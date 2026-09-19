import json, time, urllib.request, uuid

URL = "http://localhost:8000/predict"


def call(text):
    req = urllib.request.Request(
        URL,
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        body = json.load(r)
        total_ms = (time.perf_counter() - t0) * 1000
        return body["label"], r.headers["X-Cache"], float(r.headers["X-Handler-Time-Ms"]), total_ms


call("warm-up " + uuid.uuid4().hex)

miss_h, miss_t, hit_h, hit_t = [], [], [], []
for i in range(30):
    text = "Hey, are we meeting for lunch? " + uuid.uuid4().hex
    label1, c1, h1, t1 = call(text)
    assert c1 == "MISS", c1
    miss_h.append(h1); miss_t.append(t1)
    for _ in range(3):
        label2, c2, h2, t2 = call(text)
        assert c2 == "HIT" and label2 == label1
        hit_h.append(h2); hit_t.append(t2)
    if i == 0:
        print(f"example pair: MISS handler={h1:.3f} ms total={t1:.2f} ms | HIT handler={h2:.3f} ms total={t2:.2f} ms")

avg = lambda xs: sum(xs) / len(xs)
print(f"MISS  avg handler {avg(miss_h):.3f} ms | avg end-to-end {avg(miss_t):.2f} ms  (n={len(miss_h)})")
print(f"HIT   avg handler {avg(hit_h):.3f} ms | avg end-to-end {avg(hit_t):.2f} ms  (n={len(hit_h)})")