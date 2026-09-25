# Connecting a Singularity job to a remote machine via Cloudflare Tunnel

**What this is:** a way to let a **Singularity job** (which can only make *outbound* network
connections, and only to a small allow-list) reach a **TCP service running on a remote machine**
(for example a dev box that has *no public IP* and accepts no inbound connections). The
canonical use-case is exposing that machine's **Docker daemon** so a Singularity job can run
`docker` commands against it — i.e. **use the remote machine as a remote Docker host**. The
same recipe works for *any* TCP service (an HTTP API, a database, a model server, etc.).

**What this is NOT:** this is **not SSH**. We are not opening a shell. We are creating a plain
**TCP pipe** between the two machines so that a client program on the Singularity side can talk
to a server program on the remote side as if they were on the same localhost.

Throughout, replace the placeholders with your own values:

| Placeholder | Meaning | Example |
|---|---|---|
| `<DOMAIN>` | a domain (zone) you control on Cloudflare | `example.com` |
| `<HOST>` | the hostname you'll expose the service at | `docker.example.com` |
| `<TUNNEL>` | a name for your tunnel | `my-docker` |
| `<TUNNEL_UUID>` | the UUID Cloudflare assigns when you create the tunnel | `xxxxxxxx-...` |
| `<USER>` | your login on the remote machine | — |
| `<REMOTE_PORT>` | the TCP port the service listens on (remote side) | `2376` |
| `<LOCAL_PORT>` | the TCP port the client listens on (Singularity side) | `2375` |

---

## 0. Why this is needed (the constraint)

| Machine | Can it accept inbound? | Can it make outbound? |
|---|---|---|
| **Remote dev box** | no (often no public IP, NAT only) | yes (incl. HTTPS to Cloudflare) |
| **Singularity job pod** | no (multi-tenant K8s) | yes, but only to an allow-list (Cloudflare/GitHub/etc. are typically reachable) |

Neither side can be a normal "server" the other dials into directly. The trick: **both sides
dial *out* to Cloudflare's edge**, and Cloudflare stitches the two outbound connections into
one logical pipe. The remote side publishes a hostname; the Singularity side connects to that
hostname and gets a local socket wired straight to the remote service.

```
  Remote dev box                      Cloudflare edge                   Singularity job
  --------------                      ---------------                   ---------------
  service (e.g. dockerd unix sock)                                      client (docker SDK)
        |                                                                      |
  socat -> tcp://localhost:<REMOTE_PORT>                            tcp://127.0.0.1:<LOCAL_PORT>
        |                                                                      |
  cloudflared  -- outbound QUIC/TLS -->  [    <HOST>    ]  <-- outbound QUIC/TLS -- cloudflared access tcp
        (tunnel run)                     (named tunnel route)            (access tcp)
```

The Singularity job ends up with a **local port `127.0.0.1:<LOCAL_PORT>`** that behaves exactly
like the remote machine's service.

---

## 1. One-time Cloudflare setup

You need a free Cloudflare account and a domain on it (domains cost ~$10/yr; the domain is
pure plumbing, nobody types it). Do this once, on the remote machine:

```bash
# install cloudflared (Linux x86-64)
curl -fSL -o ~/.local/bin/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x ~/.local/bin/cloudflared

cloudflared tunnel login                       # opens a browser; authorize your <DOMAIN> zone
cloudflared tunnel create <TUNNEL>             # writes ~/.cloudflared/<TUNNEL_UUID>.json + prints the UUID
cloudflared tunnel route dns <TUNNEL> <HOST>   # creates the DNS CNAME <HOST> -> tunnel
```

Then create `~/.cloudflared/config.yml` mapping the hostname to your local service:

```yaml
tunnel: <TUNNEL_UUID>
credentials-file: /home/<USER>/.cloudflared/<TUNNEL_UUID>.json
ingress:
  - hostname: <HOST>
    service: tcp://localhost:<REMOTE_PORT>     # the remote-side service we expose
  - service: http_status:404
```

> Note `service: tcp://...` -- that's what makes this a **raw TCP tunnel**, not an HTTP one.
> Raw TCP is required for Docker's `exec` streams (HTTP-hijack), databases, etc.

The files needed to *run* the tunnel later are `~/.cloudflared/cert.pem`,
`~/.cloudflared/<TUNNEL_UUID>.json`, and `config.yml`. Keep them private; back them up
somewhere durable (they don't change).

---

## 2. Remote side -- publish the service

The remote machine runs **two** background processes:

1. **`socat`** -- bridges a *unix socket* to a *TCP port* (cloudflared can only forward TCP,
   not unix sockets). **Skip this if your service already listens on a TCP port.** For Docker,
   the daemon listens on `/var/run/docker.sock`, so we bridge it.
2. **`cloudflared tunnel run`** -- opens the outbound tunnel and serves the hostname.

```bash
# 1) (Docker only) expose the docker unix socket as TCP on localhost:<REMOTE_PORT>
nohup socat TCP-LISTEN:<REMOTE_PORT>,reuseaddr,fork UNIX-CONNECT:/var/run/docker.sock \
      > /tmp/socat.log 2>&1 &

# 2) start the named tunnel
nohup ~/.local/bin/cloudflared tunnel --config ~/.cloudflared/config.yml run <TUNNEL> \
      > /tmp/cf-tunnel.log 2>&1 &
```

Run these inside `tmux`/`screen` so they survive your SSH session:

```bash
tmux new -s tunnel        # start the two commands above inside; Ctrl-b d to detach
```

When `cloudflared` logs show several **registered tunnel connections** (e.g. 4 healthy edge
connections), the service is reachable at `<HOST>`.

> If the remote machine reboots, both processes die -- just re-run them. (Optional: a
> `@reboot` cron entry, see section 6.)

---

## 3. Singularity side -- connect to the service (runs inside the job)

Inside the Singularity job's command, do three things:

1. download the `cloudflared` binary (GitHub egress is allowed),
2. start `cloudflared access tcp`, which opens the outbound tunnel and **listens on a local
   port** (`127.0.0.1:<LOCAL_PORT>`),
3. point your client at `tcp://127.0.0.1:<LOCAL_PORT>`.

```bash
# 1) get cloudflared (GitHub is reachable from Singularity)
curl -fSL -o /tmp/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x /tmp/cloudflared

# 2) open the local TCP listener wired to the remote service
nohup /tmp/cloudflared access tcp \
  --hostname <HOST> \
  --url     tcp://127.0.0.1:<LOCAL_PORT> \
  > /tmp/cf-access.log 2>&1 &

# wait for the listener to come up
for i in $(seq 1 30); do
  ss -ltn 2>/dev/null | grep -q ':<LOCAL_PORT> ' && break
  sleep 1
done

# 3) now anything that speaks the service's protocol can use localhost:<LOCAL_PORT>
export DOCKER_HOST=tcp://127.0.0.1:<LOCAL_PORT>
docker version            # talks to the remote machine's dockerd
docker run --rm alpine:3.20 echo "hello from the remote docker host"
```

From here, **any tool that respects `DOCKER_HOST`** (the `docker` CLI, the Python `docker` SDK,
agent frameworks, etc.) transparently uses the remote machine's Docker daemon. The pod's own
filesystem is untouched; containers run on the remote machine.

### Using it from Python

```python
import docker
client = docker.DockerClient(base_url="tcp://127.0.0.1:<LOCAL_PORT>")
print(client.version()["Version"])
ct = client.containers.run("alpine:3.20", "sleep 30", detach=True)
print(ct.exec_run("cat /etc/os-release").output.decode())   # exec works over the tunnel
ct.stop(); ct.remove()
```

> If your tool builds the Docker URL from an "ip" argument as `tcp://<ip>:<LOCAL_PORT>`, pass
> `ip = 127.0.0.1`. Also make sure the client does **not** force TLS (the tunnel is plain TCP):
> e.g. unset / empty `DOCKER_TLS_VERIFY`.

---

## 4. Verifying it works

From the Singularity job log you should see:

```
Start Websocket listener host=127.0.0.1:<LOCAL_PORT>   # cloudflared access is up
Server Version: 29.x                                   # docker version succeeded
hello from the remote docker host                      # a container actually ran remotely
```

Round-trip latency is ~200-300 ms per call (one Cloudflare edge hop each way). That's fine for
Docker-exec-per-step agent loops, which already take minutes per trajectory.

---

## 5. Mental model / FAQ

- **Direction of dialing:** *both* machines dial OUT to Cloudflare. Neither listens for
  inbound. That's why it works despite no public IPs and despite Singularity's egress
  allow-list (Cloudflare is on the allow-list).
- **What is exposed:** exactly one TCP service per hostname (`tcp://localhost:<REMOTE_PORT>`).
  It is **not** a shell, **not** SSH, **not** the whole machine -- just that one port.
- **Who can reach it:** by default anyone who knows the hostname can reach the service. For
  real isolation add a **Cloudflare Access service-token policy** (see section 6); clients then
  must pass `--service-token-id` / `--service-token-secret`.
- **Multiple services:** add more `ingress` hostnames in `config.yml`
  (`vllm.<DOMAIN> -> tcp://localhost:8000`, etc.), route DNS for each, and on the client run
  one `cloudflared access tcp` per service pointing at a different local port.
- **Multiple remote machines (a pool):** give each its own subdomain (`docker2.<DOMAIN>`, ...),
  run a tunnel on each, and have clients pick one.

---

## 6. Optional hardening

1. **Lock down with a Cloudflare Access service token** (recommended before production):
   - Zero Trust dashboard -> Access -> Service Tokens -> create one (save Client ID + Secret).
   - Access -> Applications -> Self-hosted -> hostname `<HOST>` -> policy `Action: Service Auth`,
     `Include: Service Token = <your token>`.
   - Client side then becomes:
     ```bash
     cloudflared access tcp \
       --hostname <HOST> \
       --url tcp://127.0.0.1:<LOCAL_PORT> \
       --service-token-id "$CF_ACCESS_CLIENT_ID" \
       --service-token-secret "$CF_ACCESS_CLIENT_SECRET"
     ```
   - Without the token, requests get `HTTP 403` at the edge. Keep the secret out of code/logs
     (read it from an env var or a mode-600 file).
2. **Survive reboots** with a user crontab on the remote machine:
   ```cron
   @reboot /usr/bin/nohup /home/<USER>/.local/bin/socat TCP-LISTEN:<REMOTE_PORT>,reuseaddr,fork UNIX-CONNECT:/var/run/docker.sock >/tmp/socat.log 2>&1 &
   @reboot /usr/bin/nohup /home/<USER>/.local/bin/cloudflared tunnel --config /home/<USER>/.cloudflared/config.yml run <TUNNEL> >/tmp/cf-tunnel.log 2>&1 &
   ```
3. **Stable URL across restarts:** a *named* tunnel (what this guide uses) keeps the same
   hostname forever. A "quick tunnel" -- `cloudflared tunnel --url ...` with no name -- gets a
   new random `*.trycloudflare.com` URL every restart; only use that for throwaway tests.

---

## 7. Quick checklist

- [ ] Remote: cloudflared binary + ~/.cloudflared/{cert.pem, <TUNNEL_UUID>.json, config.yml} present
- [ ] Remote: socat bridging the unix socket to localhost:<REMOTE_PORT> (if exposing Docker)
- [ ] Remote: cloudflared tunnel run <TUNNEL> up, shows healthy edge connections
- [ ] Remote: a probe to <HOST> reaches the service
- [ ] Singularity: cloudflared access tcp --hostname <HOST> --url tcp://127.0.0.1:<LOCAL_PORT> listening
- [ ] Singularity: DOCKER_HOST=tcp://127.0.0.1:<LOCAL_PORT> docker version succeeds
