from moviepy.editor import *
from moviepy.video.tools.credits import credits1

import speech_recognition
from pypinyin import pinyin
import re
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import librosa
import librosa.display
import scipy
import json
import wave
#使用標準的格式就不會有問題
Filmpath = "jiujiu.mp4"
#Filmpath = "kzero.mp4"
#Filmpath = "./TestFilm/no_music.mp4"
#Filmpath = "./TestFilm/2019-07-09-1928-57.mp4"

def remove_punctuation(line):
	rule = re.compile("[^a-zA-Z0-9\u4e00-\u9fa5]")
	line = rule.sub('',line)
	return line

def showpreview(clip):
	#show text position and effect

	#clip.preview()
	clip.save_frame("123.png", t=2)

def settxt(clip, text, fontsize = 36, font = "", cl = "white", bgc = "transparent", opacity=0.5, pos="bottom"):
	#clip is major video
	#text is string for text
	#fontsize is size
	#cl is color
	#bgc is background color
	#TextClip.list("color") to show all color

	if bgc != "transparent":
		txt_clip = TextClip(text, fontsize= 36, color=bgc, bg_color=bgc)
		txt_clip = txt_clip.set_opacity(opacity)
		txt_clip1 = TextClip(text, fontsize= 36, color=cl)
		final_text = CompositeVideoClip([txt_clip, txt_clip1])
	else:
		final_text = TextClip(text, fontsize= 36, color=cl)
	final_text = final_text.set_pos(pos)
	final_video = CompositeVideoClip([clip, final_text])
	return final_video

def step1compare(ans_text, audio_text):
	score = 0

	for i in range(len(ans_text)):
		if ans_text[i] in audio_text:
			score = score + 1

	return score
	
def step2compare(ans_text, audio_text):
	score = 0

	for i in range(len(ans_text)):
		if ans_text[i] in audio_text:
			score = score + 1

	return score

def compare_texts(text1, text2):

	score = 0
	for i in range(len(text1)):
		if text1[i] == text2[i]:
			score = score + 1
	return score/len(text1)

def check_in(longtext, shorttext, startpoint):
	#return position
	for i in range(len(longtext) - len(shorttext) + 1):
		if i < startpoint:
			continue
		score = 0

		for j in range(len(shorttext)):
			if longtext[i+j] == shorttext[j]:
				score = score + 1
		#print(longtext[i:])
		if score/len(shorttext) >= 0.8:
			return i
	return -1

def main_algorithm(clip, subs):
	#要切割不然一次送全部的sound進去, recognize不理我

	interval_time = 20
	time_step = 0.1
	start_time = 0
	end_time = 0
	step = 0 
	prescore = 0
	score = 0

	#1. Get the Start Point
	total_audio = clip.audio
	total_duration = total_audio.duration
	total_audio.write_audiofile("123.wav")
	x_1, fs = librosa.load('123.wav')
	for i in range(len(x_1)):
		if abs(x_1[i]) > 0.01:
			start_time = i/fs
			break
	
	#Debug librosa
	#plt.figure(figsize=(16, 4))
	#librosa.display.waveplot(x_1, sr=fs)
	#plt.title('Slower Version $X_1$')
	#plt.tight_layout()
	#plt.show()
	
	#1. 確認要的字在不在裡面	
	r = speech_recognition.Recognizer()
	with speech_recognition.AudioFile("123.wav") as source:
		audio = r.record(source)
	audio_text = r.recognize_google(audio,language='zh-tw')
	pos_start = 0
	pos_end = 0
	for i in range(len(subs)):
		pos_start = check_in(audio_text, subs[i].text[:5], pos_end)
		pos_end = check_in(audio_text, subs[i].text[-5:], pos_start)
		subs[i].comparision = audio_text[pos_start:pos_end+5]

		#print(pos_start, pos_end)
		#print(subs[i].text)
		#print(subs[i].comparision)
		#print(audio_text[pos_start:pos_end+5])

	#2. 嚴格比對comparision與剪輯下來的字
	
	pos_start = 0
	pos_end = 0
	for i in range(len(subs)):
		end_time = start_time + len(subs[i].text)*0.3
		#校對前面
		while True:
			asub = total_audio.subclip(start_time, end_time)
			asub.write_audiofile("123.wav")
			r = speech_recognition.Recognizer()
			with speech_recognition.AudioFile("123.wav") as source:
				audio = r.record(source)
			audio_text = r.recognize_google(audio,language='zh-tw')
		
			if audio_text == subs[i].comparision:
				break
			#print(audio_text)
			#print(subs[i].comparision)
			pos_start = check_in(audio_text, subs[i].comparision[:5], 0)
			#print(pos_start)
			if pos_start == -1:
				start_time = start_time - time_step
			elif pos_start > 0:
				start_time = start_time + time_step * pos_start
			elif pos_start == 0:
				subs[i].start_time = start_time
				break

		print(subs[i].start_time)
		#校對後面
		while True:
			asub = total_audio.subclip(start_time, end_time)
			asub.write_audiofile("123.wav")
			r = speech_recognition.Recognizer()
			with speech_recognition.AudioFile("123.wav") as source:
				audio = r.record(source)
			audio_text = r.recognize_google(audio,language='zh-tw')
		
			if audio_text == subs[i].comparision:
				break
			#print(len(audio_text), len(subs[i].comparision), interval_time * (len(audio_text) - len(subs[i].comparision)))
			print(audio_text)
			print(subs[i].comparision)
			pos_end = check_in(audio_text, subs[i].comparision[-5:], 0)
			if pos_end == -1:
				end_time = end_time + time_step * (-len(audio_text) + len(subs[i].comparision))
			elif pos_end == len(audio_text) - 5:
				subs[i].end_time = end_time
				break
			else:
				end_time = end_time + time_step
			#print(pos_end, start_time, end_time)
		print(subs[i].end_time)
		start_time = end_time - 10*time_step

		
			


		#print(start_time, end_time)
	#First Step : Compare Head
	"""while True:
		
		r = speech_recognition.Recognizer()
		with speech_recognition.AudioFile("123.wav") as source:
			audio = r.record(source)
		audio_text = r.recognize_google(audio,language='zh-tw')
		score_s = compare_texts(audio_text[:5], sub.text[:5])
		score_e = compare_texts(audio_text[-5:], sub.text[-5:])
		print(score_s, score_e, start_time, end_time)
		if score_s >= 0.8 and score_e >= 0.8:
			break
		else:
			if score_s < 0.8:
				start_time = start_time - time_step
			if score_e < 0.8:
				end_time = end_time - time_step"""

	
		
				

	
	
	#First Step : Rough Compare 30s Interval
	#Second Step : forward start_time to minimize interval
	#Third Step : backward end_time to minimize interval	
	"""while True:
		
		if subidx == len(subs):
			break
		if subidx == 0:
			if step == 0:
				end_time = end_time + interval_time
				step = 1
			elif step == 1:
				start_time = start_time + time_step
			elif step == 2:
				if (abs(len(audio_text) - len(sub_text))) > 5:
					end_time = end_time - time_step * (abs(len(audio_text) - len(sub_text)))
				else:
					end_time = end_time - time_step
				
		else:
			if step == 0:
				start_time = subs[subidx-1].end_time
				end_time = start_time + interval_time
				step = 2
			#elif step == 1:
			#	start_time = start_time + 1
			elif step == 2:
				end_time = end_time - time_step

		if end_time > total_duration:
			end_time = total_duration

		#print(start_time, end_time)
		asub = total_audio.subclip(start_time, end_time)
		asub.write_audiofile("123.wav")
		r = speech_recognition.Recognizer()
		with speech_recognition.AudioFile("123.wav") as source:
			audio = r.record(source)
		audio_text = r.recognize_google(audio,language='zh-tw')
		
		sub_text = subs[subidx].text
		if step > 0:
			score = step1compare(sub_text, audio_text)
			if score < len(sub_text)*0.5:
				start_time = start_time + interval_time
			elif step == 1:
				if score < prescore:
					start_time = start_time - time_step
					step = 2
				else:
					prescore = score
			elif step == 2:
				if score < prescore and abs(len(audio_text) - len(sub_text)) < 5:
					end_time = end_time + time_step				
					subs[subidx].start_time = start_time
					subs[subidx].end_time = end_time
					step = 0
					prescore = 0
					score = 0
					subidx = subidx + 1
				else:
					prescore = score
		print(subidx, score, prescore, start_time, end_time)
		print(audio_text)
		print(sub_text)
		#elif step == 2:"""



def extract_vocal(path):
	#print(path)
	y, sr = librosa.load(path)
	#print(y)
	S_full, phase = librosa.magphase(librosa.stft(y))
	#print(S_full)
	idx = slice(*librosa.time_to_frames([30, 35], sr=sr))
	S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))
	S_filter = np.minimum(S_full, S_filter)

	margin_i, margin_v = 2, 10
	power = 2

	mask_i = librosa.util.softmask(S_filter,
	                               margin_i * (S_full - S_filter),
	                               power=power)

	mask_v = librosa.util.softmask(S_full - S_filter,
	                               margin_v * S_filter,
	                               power=power)
	S_foreground = mask_v * S_full
	S_background = mask_i * S_full
	D_foreground = S_foreground * phase
	y_foreground = librosa.istft(D_foreground)

	D_background = S_background * phase
	y_background = librosa.istft(D_background)
	#print(y_foreground)
	
	maxv = np.iinfo(np.int16).max
	scipy.io.wavfile.write("foreground.wav", sr, (y_foreground* maxv).astype(np.int16))
	#librosa.output.write_wav("foreground.wav", (y_foreground* maxv).astype(np.int16), sr)
	#librosa.output.write_wav("background.wav", (y_background* maxv).astype(np.int16), sr)





class subtitle():
	def __init__(self, sno, text):
		self.serialnum = sno
		self.text = text
		self.start_time = 0
		self.end_time = 0
		self.done = 0
		self.comparision = ""

def load_text(textpath):
	f = open(textpath)
	text = []
	for buf in f:
		if len(buf.replace("\n","")) == 0:
			continue
		text.append(buf.replace("\n",""))
	f.close()
	return text

def get_subs(path):
	text = load_text(path)
	subs = []
	for i in range(len(text)):
		subs.append(subtitle(i, remove_punctuation(text[i])))
	return subs

def parsing_audio():
	"""
		nchannels : 1 mono 2 stereo
		sampwidth : 代表一個採用點用幾個byte 通常1~2個byte
		framerate : 44100 因為人耳可以聽到20k, 所以要用兩倍的頻率去採樣
		nframes : 可以表示長度
		nframes / framerate = 音訊長度

		y, sr = librosa.load(path, sr=None) sr=None才可以讀取44100 否則是22050
		y是float
		要轉換成int 要乘 maxv = np.iinfo(np.int16).max 32767
		stft short time fourier transform 短時距傅立葉變換是傅立葉變換的一種變形
		計算短時距傅立葉變換(STFT)的過程是將長時間訊號分成數個較短的等長訊號，
		然後再分別計算每個較短段的傅立葉轉換。通常拿來描繪頻域與時域上的變化，為時頻分析中其中一個重要的工具
		magphase 把 訊號拆成 magnitude, phase
		stft shape = shape=(1 + n_fft/2, t) n_fft預設2048 
		win_length = n_fft (<= n_fft) 音窗
		hop_length : win_length / 4 STFT column
		
		For a 44100 sampling rate, we have a 22050 Hz band. With a 1024 FFT size, we divide this band into 512 bins.
		FR = 22050/1024 ≃ 21,53 Hz.
	"""
	import matplotlib.pyplot as plt
	y, sr = librosa.load("./Audio/0.wav", sr=None)
	#idx = slice(*librosa.time_to_frames([30, 35], sr=sr))
	#print(idx, *librosa.time_to_frames([30, 35]))
	# And compute the spectrogram magnitude and phase
	a = librosa.stft(y)
	S_full, phase = librosa.magphase(librosa.stft(y))
	print(a.shape, S_full.shape, phase.shape, y.shape				)
	"""plt.figure(figsize=(12, 4))
	librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
	                         y_axis='log', x_axis='time', sr=sr)
	plt.colorbar()
	plt.tight_layout()
	plt.show()"""

	#print(S_full, phase)
	"""fw = wave.open("./Audio/total.wav",'r')
	params = fw.getparams()
	nchannels, sampwidth, framerate, nframes = params[:4]
	strData = fw.readframes(nframes)
	waveData = np.fromstring(strData, dtype=np.int16)
	waveData = waveData*1.0/max(abs(waveData))  # normalization 轉換成float

	#print(waveData)
	fw.close()"""

if __name__ == "__main__":
	#需要的功能
	#endpoint
	#語音辨識
	#強化人聲

	get_subs("./TestFilm/原文.txt")

	#Video clip Audtio
	clip = VideoFileClip(Filmpath)
	total_duration = clip.duration
	clip_audio = clip.audio.subclip(0, 30)
	clip_audio.write_audiofile("./Audio/0.wav")

	parsing_audio()

	"""i = 0
	step = 10
	startt = 0
	endt = step
	while True:
		print(i)
		if endt > total_duration:
			endt = total_duration
		asub = clip_audio.subclip(startt, endt)
		asub.write_audiofile("./audio/" + str(i) + ".wav")
		#print("Extract")
		#extract_vocal("./audio/" + str(i) + ".wav")
		
		#print("Extract finish")
		r = speech_recognition.Recognizer()
		with speech_recognition.AudioFile("./audio/" + str(i) + ".wav") as source:
			#r.adjust_for_ambient_noise(source)
			audio = r.record(source)

		#f = open("./Google_Key/My First Project-1a69a44f2967.json", "r")
		#jdata = f.read()
		#audio_text = r.recognize_google_cloud(audio, credentials_json=jdata, language='zh-tw')
		try:
			audio_text2 = r.recognize_google(audio, language='zh-tw')
		except:
			startt = startt + step
			endt = endt + step
			if startt > total_duration:
				break
			i = i + 1
			continue
		#print(audio_text)
		print(audio_text2)
		break
		startt = startt + step
		endt = endt + step
		if startt > total_duration:
			break
		i = i + 1
	#for i in range(len(subs)):
	#main_algorithm(clip, subs)"""
#showpreview(clip)

##########a = clip.audio
##########video_time = clip.duration
##########i = 0
##########while i + 30 < video_time:
##########	
##########	clip1 = clip.subclip(i*10, (i+1)*10)
##########	a = clip1.audio
##########	print(a.duration)
##########	a.write_audiofile("123.wav")
##########	try:
##########		with speech_recognition.AudioFile("123.wav") as source:
##########			
##########			audio = r.record(source)
##########			text = r.recognize_google(audio,language='zh-tw')#, show_all=True)
##########			
##########			print(text)
##########	except:
##########		pass
##########	#os.remove("123.wav")
##########	i = i + 1
	
#clip = clip.resize((1440,960))
#print(clip.duration)
#print(dir(clip))
#print(clip.h, clip.w)

#txt_clip = TextClip("r23tg3g573", fontsize= 36, color="white")
#txt_clip = txt_clip.set_pos("bottom").set_duration(10)
#video = CompositeVideoClip([clip, txt_clip])
#video.write_videofile("123.mp4")
           #.subclip(37,46)
           #.speedx( 0.4)
           #.fx( vfx.colorx, 0.7))
 