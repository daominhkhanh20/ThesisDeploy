VERSION="8"
export KAGGLE_USERNAME='daominhkhanh'
export KAGGLE_KEY="53d2021e6812290870fc3520cbeee5ea"
kaggle datasets download -d daominhkhanh/model-compile-v$VERSION
unzip model-compile-v$VERSION.zip
mv qa/model.pt model/qa_model/1
rm -r qa
rm model-compile-v$VERSION.zip