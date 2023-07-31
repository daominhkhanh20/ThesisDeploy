VERSION="5"
export KAGGLE_USERNAME='daominhkhanh'
export KAGGLE_KEY="53d2021e6812290870fc3520cbeee5ea"
kaggle datasets download -d daominhkhanh/model-compile-v$VERSION
unzip model-compile-v$VERSION.zip
mkdir model/sbert_retrieval/1
mkdir model/qa_model/1
mv sbert/model.pt model/sbert_retrieval/1
mv qa/model.pt model/qa_model/1
mv corpus.json model/corpus.json
mv dev.tar.gz model
rm -r sbert
rm -r qa
rm model-compile-v$VERSION.zip
