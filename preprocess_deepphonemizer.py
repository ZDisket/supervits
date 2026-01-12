import argparse
import re
from tqdm import tqdm
from utils import load_filepaths_and_text
from dp.phonemizer import Phonemizer

def collapse_whitespace(text):
  return re.sub(r'\s+', ' ', text)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_extension", default="cleaned")
  parser.add_argument("--text_index", default=1, type=int)
  parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])
  parser.add_argument("--checkpoint", default="en_us_cmudict_ipa_forward.pt")
  parser.add_argument("--lang", default="en_us")

  args = parser.parse_args()
    
  print(f"Loading Phonemizer from {args.checkpoint}...")
  # Initialize the phonemizer as requested
  phonemizer = Phonemizer.from_checkpoint(args.checkpoint)
  phonemizer.lang_phoneme_dict = None

  for filelist in args.filelists:
    print("START:", filelist)
    filepaths_and_text = load_filepaths_and_text(filelist)
    new_filelist = filelist + "." + args.out_extension
    
    # Collect all text entries
    texts = [entry[args.text_index] for entry in filepaths_and_text]
    
    print(f"Phonemizing {len(texts)} entries...")
    try:
      # Feed the entire list to the phonemizer for faster processing
      cleaned_texts = phonemizer(texts, lang=args.lang)
      # Collapse whitespace for each entry
      cleaned_texts = [collapse_whitespace(ph) for ph in cleaned_texts]
    except Exception as e:
      print(f"ERROR: Phonemization failed for {filelist}: {e}")
      continue

    print(f"Saving to {new_filelist}...")
    with open(new_filelist, "w", encoding="utf-8") as f:
      for i, cleaned_text in enumerate(cleaned_texts):
        filepaths_and_text[i][args.text_index] = cleaned_text
        f.write("|".join(filepaths_and_text[i]) + "\n")
  
  print("Done!")
