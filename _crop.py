from PIL import Image
import os
os.makedirs('_crops', exist_ok=True)
for fn in ['1.png','2.png']:
    im = Image.open(fn).convert('RGB')
    w,h = im.size
    parts = {'top':(0,0,w,int(h*0.40)),'mid':(0,int(h*0.33),w,int(h*0.72)),'bot':(0,int(h*0.66),w,h)}
    for name,box in parts.items():
        c = im.crop(box)
        c = c.resize((c.width*3, c.height*3), Image.LANCZOS)
        out=f'_crops/{fn[0]}_{name}.png'
        c.save(out)
        print(out, c.size)
