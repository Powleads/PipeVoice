import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from wisprlite import agent_bridge, mcp_shim

def test_send_forwards_to_bridge():
    listener = agent_bridge.ControlListener(0, lambda req: {"status": "ok", "echo": req})
    listener.start()
    try:
        import wisprlite.mcp_shim as shim
        shim._port = lambda: listener.port
        resp = shim._send("transcribe", path="/x", format="json")
        assert resp["status"] == "ok", resp
        assert resp["echo"] == {"op": "transcribe", "path": "/x", "format": "json"}, resp
    finally:
        listener.stop()

def test_send_app_not_running():
    import wisprlite.mcp_shim as shim
    shim._port = lambda: 1
    resp = shim._send("listen")
    assert resp["status"] == "app_not_running", resp

if __name__ == "__main__":
    test_send_forwards_to_bridge(); test_send_app_not_running()
    print("OK")
