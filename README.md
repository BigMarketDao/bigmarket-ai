# bigmarket.ai

Python server manages prompts and tools and communicates with LLM layer.

## Developing

```bash
source ~/bigmarket-ai/bin/activate
pip install requirements
python server.py
```

## Python version

```bash
python --version
Python 3.12.9
```

## Service management

```bash
sudo vi /etc/systemd/system/bigmarket-ai.service
sudo systemctl daemon-reload
sudo systemctl restart bigmarket-ai
sudo systemctl status bigmarket-ai
journalctl -u bigmarket-ai --no-pager --lines=50 | tail -n 20
```
