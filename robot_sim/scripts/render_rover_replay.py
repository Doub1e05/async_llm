"""Encode actual Gazebo follow-camera frames with clearly labelled test telemetry."""
import argparse
import bisect
import json
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont


def render(directory):
    summary=json.loads((directory/"summary.json").read_text())
    frames=[json.loads(line) for line in (directory/"frames.jsonl").read_text().splitlines()]
    if not frames:raise ValueError("No recorded Gazebo camera frames")
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font=ImageFont.truetype(font_path,19)
    small=ImageFont.truetype(font_path,16)
    title=ImageFont.truetype(font_path,24)
    stamps=[f["sim_time"] for f in frames]
    fps=3
    command=["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24",
             "-s","960x600","-r",str(fps),"-i","-","-an","-c:v","libx264","-threads","1",
             "-preset","veryfast","-crf","22","-pix_fmt","yuv420p","-movflags","+faststart",
             str(directory/"replay.mp4")]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        total=max(1,int((stamps[-1]-stamps[0])*fps))
        for index in range(total+fps*3):
            stamp=min(stamps[-1],stamps[0]+index/fps)
            frame=frames[max(0,bisect.bisect_right(stamps,stamp)-1)]
            canvas=Image.new("RGB",(960,600),"#101923")
            draw=ImageDraw.Draw(canvas)
            draw.text((20,12),"Yandex mini rover / Gazebo Harmonic",font=title,fill="white")
            draw.text((20,45),f"Scripted physics test: {summary['case']} / {stamp-stamps[0]:.1f} simulation seconds",font=font,fill="#92d9cc")
            with Image.open(directory/frame["file"]) as image:
                canvas.paste(image,(0,85))
            draw.text((660,100),"ACTUAL GAZEBO",font=font,fill="white")
            draw.text((660,128),"FOLLOW CAMERA",font=font,fill="white")
            draw.text((660,175),"Phase: "+frame["phase"],font=small,fill="#ffcf85")
            draw.text((660,220),"Mass: 0.500 kg",font=small,fill="white")
            draw.text((660,250),"Track: 115.36 mm",font=small,fill="white")
            draw.text((660,280),"Motor cap: 0.03 Nm",font=small,fill="white")
            draw.text((660,330),"1 CPU / 0 GPU",font=small,fill="#92d9cc")
            draw.text((660,370),"Estimated physics;",font=small,fill="#b8c5d1")
            draw.text((660,395),"not hardware calibration",font=small,fill="#b8c5d1")
            if index>=total:
                draw.rectangle((650,440,950,530),fill="#24333f")
                draw.text((660,450),"TEST "+("PASS" if summary["passed"] else "FAIL"),font=title,fill="#92d9cc" if summary["passed"] else "#ffaea0")
                draw.text((660,490),"Final card / motion ended",font=small,fill="white")
            if index==0:canvas.save(directory/"preview.png")
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close()
    if process.wait()!=0:raise RuntimeError("Video encoding failed")
    return {"file":str(directory/"replay.mp4"),"frames":len(frames)}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    print(json.dumps(render(parser.parse_args().directory)))
