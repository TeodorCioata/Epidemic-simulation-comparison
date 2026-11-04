"""
This file is for generating side-by-side HTML animations of models M1 and M2 for the epidemic simulations.
The animation includes a button for stopping and restarting the animation.
"""
import json

def build_comparison_html(m1_frames, m2_frames, times, frame_ms):
    """
    This function generates synchronized animations for M1 and M2.
    Show both models side by side. Use a shared play/stop button and a shared time slider.
    Return the HTML page as a string.
    """
    payload = json.dumps({"m1": m1_frames, "m2": m2_frames, "times": times, "interval": frame_ms})
    return """<!doctype html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <!--Define the style of the animation-->
                    <style>
                        body {
                            font-family: sans-serif;
                            margin: 0;
                            color: #222
                        }
                
                        .controls {
                            display: flex;
                            gap: 12px;
                            align-items: center;
                            padding: 8px 4px
                        }
                
                        #slider {
                            flex: 1;
                            min-width: 60px
                        }
                
                        .panels {
                            display: grid;
                            grid-template-columns: 1fr 1fr;
                            gap: 8px
                        }
                
                        .panel {
                            text-align: center;
                            min-width: 0
                        }
                
                        .panel img {
                            width: 100%;
                            height: auto;
                            display: block
                        }
                
                        button {
                            padding: 6px 14px;
                            cursor: pointer
                        }
                
                        #time {
                            white-space: nowrap;
                            font-variant-numeric: tabular-nums
                        }
                    </style>
                </head>
                <body>
                    <div class="controls">
                        <button id="play">Pause</button>
                        <input id="slider" type="range" min="0" max="__LAST__" value="0">
                        <span id="time">t = 0.0</span>
                    </div>
                
                    <div class="panels">
                        <div class="panel">
                            <strong>M1 Unweighted</strong>
                            <img id="m1" alt="M1 epidemic">
                        </div>
                        <div class="panel">
                            <strong>M2 Weighted</strong>
                            <img id="m2" alt="M2 epidemic">
                        </div>
                    </div>
                
                    <script>
                        const data = __DATA__;
                        const slider = document.getElementById('slider');
                        const play = document.getElementById('play');
                
                        let index = 0, timer = null;
                
                        //Show the frames of both models at the same simulation time
                        function show(i) {
                            index = i;
                            document.getElementById('m1').src = data.m1[i];
                            document.getElementById('m2').src = data.m2[i];
                            slider.value = i;
                            document.getElementById('time').textContent = 't = ' + data.times[i].toFixed(1);
                        }
                
                        //Stop the animation and update the label of the stop button
                        function stop() {
                            if (timer !== null)
                                clearInterval(timer);
                            timer = null;
                            play.textContent = 'Play';
                        }
                
                        //Start the animation and go through the frames
                        function start() {
                            if (index === data.times.length - 1)
                                show(0);
                            stop();
                            play.textContent = 'Pause';
                            timer = setInterval(() => {
                                if (index === data.times.length - 1) {
                                    stop();
                                    return;
                                }
                                show(index + 1);
                            }, data.interval);
                        }
                        
                        //Connect the play and the pause button to the animation
                        play.addEventListener('click', () => timer === null ? start() : stop());
                
                        slider.addEventListener('input', () => {
                            stop();
                            show(Number(slider.value));
                        });
                
                        show(0);
                        start();
                    </script>
                </body>
                </html>""".replace('__LAST__', str(len(times)-1)).replace('__DATA__',payload)
