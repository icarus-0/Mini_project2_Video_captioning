import FrameCaption
import warnings
warnings. filterwarnings("ignore")

caption_count, caption_list = FrameCaption.getFrameCaptions("D:\\Projects_Code\\Miniproject2\\2.avi")

#print(caption_count)
c = 0
max_occr_caption = ""

for i in caption_list:
    if caption_count[i] > c:
        max_occr_caption = i



print("\n\n\n\n")
print(max_occr_caption)