from ...modules.runtime import log as print, add_runtime_arguments, experiment_dir
from PIL import Image
from torchvision import transforms
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Configuration")
    parser.add_argument(
        "--exp", type=str, default='',
    )
    parser.add_argument(
        "--subj", type=int, default=1, choices=[1, 2, 3],
        help="Validate on which subject?",
    )
    add_runtime_arguments(parser)
    args = parser.parse_args()


    device = "cuda" if torch.cuda.is_available() else "cpu"



    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", revision="51572668da0eb669e01a189dc22abe6088589a24", cache_dir = args.cache_dir)
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", revision="51572668da0eb669e01a189dc22abe6088589a24", cache_dir = args.cache_dir, torch_dtype=torch.float16
    )
    # The pinned checkpoint predates the image-token metadata required by Transformers 5.
    processor.num_query_tokens = model.config.num_query_tokens
    model.config.image_token_index = processor.tokenizer.convert_tokens_to_ids('<image>')
    if len(processor.tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(processor.tokenizer), mean_resizing = False)
    model.to(device)

    images = torch.load(f'{experiment_dir(args)}/frames_generated/video_subj0{args.subj}_all_recons.pt', map_location='cpu', weights_only = False)
    print(images.shape)
    all_predcaptions = []
    for i in range(images.shape[0]):
        print(i)
        x = images[i]
        # print(f"\033[92m xx {x.shape} \033[0m")

        x = transforms.ToPILImage()(x)

        # print(f"\033[92m {np.array(x).shape} \033[0m")

        inputs = processor(images=x, return_tensors="pt").to(device, torch.float16)

        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        all_predcaptions = np.hstack((all_predcaptions, generated_text))
        print(generated_text)

    torch.save(all_predcaptions, f'{experiment_dir(args)}/frames_generated/pred_test_caption.pt')

    print(f"\033[92m Saved to {experiment_dir(args)}/frames_generated/pred_test_caption.pt \033[0m")

