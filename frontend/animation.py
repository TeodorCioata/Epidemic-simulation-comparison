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

def build_comparison_html(m1_frames, m2_frames, times, frame_ms):
    """One playback clock and slider for both model animations."""
    if len(m1_frames) != len(m2_frames) or len(times) != len(m1_frames):
        raise ValueError("Both models must share the same frame times")
    payload = json.dumps({"m1": m1_frames, "m2": m2_frames, "times": times, "interval": frame_ms})
    return """<!doctype html><html><head><meta charset="utf-8"><style>
    body{font-family:sans-serif;margin:0;color:#222} .controls{display:flex;gap:12px;align-items:center;padding:8px 4px}
    #slider{flex:1;min-width:60px} .panels{display:grid;grid-template-columns:1fr 1fr;gap:8px}
    .panel{text-align:center;min-width:0} .panel img{width:100%;height:auto;display:block}
    button{padding:6px 14px;cursor:pointer} #time{white-space:nowrap;font-variant-numeric:tabular-nums}
    </style></head><body>
    <div class="controls"><button id="play">Pause</button><input id="slider" type="range" min="0" max="__LAST__" value="0"><span id="time">t = 0.0</span></div>
    <div class="panels"><div class="panel"><strong>M1 Unweighted</strong><img id="m1" alt="M1 epidemic"></div>
    <div class="panel"><strong>M2 Weighted</strong><img id="m2" alt="M2 epidemic"></div></div>
    <script>
    const data = __DATA__; const slider=document.getElementById('slider'); const play=document.getElementById('play');
    let index=0, timer=null;
    function show(i){index=i; document.getElementById('m1').src=data.m1[i]; document.getElementById('m2').src=data.m2[i]; slider.value=i; document.getElementById('time').textContent='t = '+data.times[i].toFixed(1);}
    function stop(){if(timer!==null)clearInterval(timer);timer=null;play.textContent='Play';}
    function start(){if(index===data.times.length-1)show(0); stop();play.textContent='Pause';timer=setInterval(()=>{if(index===data.times.length-1){stop();return;}show(index+1);},data.interval);}
    play.addEventListener('click',()=>timer===null?start():stop());slider.addEventListener('input',()=>{stop();show(Number(slider.value));});
    show(0);start();
    </script></body></html>""".replace('__LAST__', str(len(times)-1)).replace('__DATA__',payload)
