export KAGGLE_USERNAME='daominhkhanh'
export KAGGLE_KEY="53d2021e6812290870fc3520cbeee5ea"
kaggle datasets download -d daominhkhanh/model-compile-v2
unzip model-compile-v2.zip
mv sbert/model.pt model/sbert_retrieval/1
mv qa/model.pt model/qa_model/1
rm -r sbert
rm -r qa