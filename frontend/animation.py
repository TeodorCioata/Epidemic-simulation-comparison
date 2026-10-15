import json

PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">

<style>
  #controls {display: flex; align-items: center; gap: 12px; padding: 4px 2px 10px 2px;}
  #play {font-size: 14px; padding: 6px 12px; cursor: pointer; min-width: 96px;}
  #slider {flex: 1; cursor: pointer;}
  #label {min-width: 80px; text-align: right; font-variant-numeric: tabular-nums;}
  #frame {display: block; margin: 0 auto; max-width: 100%; height: auto;}
</style>
</head>

<body>
<div id="controls">
  <button id="play">Pause</button>
  <input id="slider" type="range" min="0" max="__LAST_INDEX__" value="0" step="1">
  <span id="label">t = 0.0</span>
</div>
<img id="frame" alt="Epidemic network">

<script>
  const frames = __FRAMES__;
  const times = __TIMES__;
  const FRAME_MS = __FRAME_MS__;
  const img = document.getElementById("frame");
  const slider = document.getElementById("slider");
  const label = document.getElementById("label");
  const playButton = document.getElementById("play");
  let index = 0;
  let timer = null;
  function show(i) {
    index = i;
    img.src = frames[i];
    slider.value = i;
    label.textContent = "t = " + times[i].toFixed(1);
  }

  function stop() {
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
    playButton.innerHTML = "Play";
  }

  function play() {
    if (index >= frames.length - 1) {
      show(0);
    }
    playButton.innerHTML = "Pause";
    timer = setInterval(() => {
      if (index >= frames.length - 1) {
        stop();
        return;
      }
      show(index + 1);
    }, FRAME_MS);
  }

  playButton.addEventListener("click", () => {
    if (timer === null) {play();} else {stop();}
  });

  slider.addEventListener("input", () => {
    stop();
    show(parseInt(slider.value, 10));
  });

  show(0);
  play();
</script>
</body>
</html>"""


def build_animation_html(frames, times, frame_ms):
    page = PAGE_TEMPLATE
    page = page.replace("__LAST_INDEX__", str(len(frames) - 1))
    page = page.replace("__FRAME_MS__", str(frame_ms))
    page = page.replace("__TIMES__", json.dumps(times))
    page = page.replace("__FRAMES__", json.dumps(frames))
    return page