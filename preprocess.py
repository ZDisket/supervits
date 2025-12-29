import argparse
import text
from tqdm import tqdm
from utils import load_filepaths_and_text

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_extension", default="cleaned")
  parser.add_argument("--text_index", default=1, type=int)
  parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])
  parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])

  args = parser.parse_args()
    

  for filelist in args.filelists:
    print("START:", filelist)
    filepaths_and_text = load_filepaths_and_text(filelist)
    new_filelist = filelist + "." + args.out_extension
    
    with open(new_filelist, "w", encoding="utf-8") as f:
      for i in tqdm(range(len(filepaths_and_text)), desc=f"Cleaning {filelist}"):
        original_text = filepaths_and_text[i][args.text_index]
        try:
          cleaned_text = text._clean_text(original_text, args.text_cleaners)
          filepaths_and_text[i][args.text_index] = cleaned_text
          f.write("|".join(filepaths_and_text[i]) + "\n")
        except KeyboardInterrupt:
          break
        except Exception as e:
          print(f"WARNING: Error cleaning text in {filelist} at index {i}: {e}")
          continue
