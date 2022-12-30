import os
import datetime
import pathlib

import torch
import whisper
import pandas as pd
import mplfinance as mpf
from pytube import YouTube

import ffmpeg
def main():
    parent_dir = pathlib.Path("C:/Users/lllang/Desktop/side_projects/openai_sandbox")
    data_dir = parent_dir / "data"
    print(data_dir)
    print(f"Is there a GPU : {torch.cuda.is_available()}")
    model = whisper.load_model('base',device = 'cuda:0')
    
    youtube_video_url = "https://www.youtube.com/watch?v=NT2H9iyd-ms"
    youtube_video = YouTube(youtube_video_url)

    print(f"Title : {youtube_video.title}")

    for stream in youtube_video.streams:
        print(stream)

    print(dir(youtube_video))

    stream = youtube_video.streams.first()

    filename = data_dir / 'fed_meeting.mp4'
    if not os.path.exists(filename):
        stream.download(filename=filename)


    in_file = filename
    out_file = data_dir / 'fed_meeting_trimmed.mp4'
    if not os.path.exists(out_file):
        print('Trimming Video')
        input_stream = ffmpeg.input(in_file)
        print(dir(ffmpeg))

        video = input_stream.trim(start = 378,end = 2715)
        audio = (input_stream
                .filter_("atrim", start = 378,end = 2715))
        videa_audio = ffmpeg.concat(video,audio,v=1,a=1)
        stream = ffmpeg.output(videa_audio, filename = out_file, format="mp4")
        ffmpeg.run(stream)
    # Runs commands from command line
    # os.system(f"ffmpeg -ss 378 -i {in_file} -t 2715 {out_file}")

    # Save a timestamp before transcription
    t1 = datetime.datetime.now()

    print(str(out_file))
    # do the transcription
    output = model.transcribe(str(out_file))

    # show time elapsed after transcription is complete
    t2 = datetime.datetime.now()
    print(f"ended at {t2}")
    print(f"time elapsed: {t2-t1}")
    print(output)

    df_spy = pd.read_csv(data_dir / "spy.csv")
    print(df_spy.head())


    for segment in output['segments']:
        second = int(segment['start'])
        second = second - (second % 5)
        df_spy.loc[second / 5 , 'text'] = segment['text']
    
    print(df_spy.head())

    df_spy['percent'] = ( (df_spy['close'] - df_spy['open']) / df_spy['open']) * 100

    big_downmoves = df_spy[df_spy.percent < -0.2]
    print(big_downmoves )


    df = df_spy
    df.index = pd.DatetimeIndex(df['date'])

    mpf.plot(df['2022-11-02 14:36':'2022-11-02 14:39'], type='candle')

    print(df_spy[50:70])
if __name__ == '__main__':
    main()