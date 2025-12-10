import datetime

def log(module, msg):
    t = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{t}][{module}] {msg}", flush=True)
