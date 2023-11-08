import argparse
import os.path
from natsort import natsorted

import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.conversation import conv_templates

from PIL import Image
Image.MAX_IMAGE_PIXELS = 16384 * 16384
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import pandas as pd
import glob
from tqdm import tqdm

def generate_texts(args, paths, model, tokenizer, image_processor, prompt, start_index, end_index):
    file_names=[]
    texts=[]
    image_tensors=[]
    for path in paths[start_index:end_index]:
        image = Image.open(path)
        # Create a new white background image
        background = Image.new('RGBA', image.size, (255, 255, 255))
        image = image.convert("RGBA")
        # Paste the image onto the background using the alpha channel as a mask
        background.paste(image, (0, 0), image)

        image=background.convert("RGB")
        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, args)
        image_tensors.append(image_tensor)

    image_tensor = torch.cat(image_tensors, 0)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    input_ids = torch.repeat_interleave(input_ids, repeats=image_tensor.size()[0], dim=0)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens
        )

    for a in range(output_ids.size()[0]):
        outputs = tokenizer.decode(output_ids[a, input_ids.shape[1]:],skip_special_tokens=True).strip()
        texts.append(outputs)
        file_names.append(os.path.basename(paths[start_index+a]))

    if args.debug:
        print("\n", {"outputs": outputs}, "\n")

    return file_names,texts

def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()

    paths = glob.glob(os.path.join(args.image_folder,"*.png"))
    paths += glob.glob(os.path.join(args.image_folder,"*.jpg"))
    paths += glob.glob(os.path.join(args.image_folder,"*.jpeg"))
    paths += glob.glob(os.path.join(args.image_folder,"*.webp"))

    paths = natsorted(paths)

    # remove broken image
    #for path in tqdm(paths):
    #    try:
    #        Image.open(path)
    #    except:
    #        os.remove(path)

    # first message
    if model.config.mm_use_im_start_end:
        user_input = DEFAULT_IM_START_TOKEN+DEFAULT_IMAGE_TOKEN+DEFAULT_IM_END_TOKEN+"\n"+args.user_prompt
    else:
        user_input = DEFAULT_IMAGE_TOKEN+"\n"+args.user_prompt

    conv.append_message(conv.roles[0], user_input)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    bs=args.batch_size
    total_batch=len(paths)//bs

    results={"file_name":[],"text":[]}
    for i in tqdm(range(total_batch)):
        file_names, texts = generate_texts(args,paths,model,tokenizer,image_processor,prompt,i*bs,(i+1)*bs)
        results["file_name"].extend(file_names)
        results["text"].extend(texts)
        if(i%args.save_every_n_steps==0):
            pd.DataFrame(results).to_csv(args.output_csv, index=False)

    if(len(paths)%bs!=0):
        file_names, texts = generate_texts(args,paths,model,tokenizer,image_processor,prompt,total_batch*bs,len(paths))
        results["file_name"].extend(file_names)
        results["text"].extend(texts)

    pd.DataFrame(results).to_csv(args.output_csv,index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--user-prompt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--save-every-n-steps", type=int, default=100)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)