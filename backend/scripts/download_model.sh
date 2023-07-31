rm -r ThesisDeploy
git clone https://github.com/daominhkhanh20/ThesisDeploy.git
cd ThesisDeploy
git checkout develop_pipeline
conda activate dev
VERSION="5"
export KAGGLE_USERNAME='daominhkhanh'
export KAGGLE_KEY="53d2021e6812290870fc3520cbeee5ea"
kaggle datasets download -d daominhkhanh/model-compile-v$VERSION
unzip model-compile-v$VERSION.zip
mkdir backend/model/sbert_retrieval/1
mkdir backend/model/qa_model/1
mv sbert/model.pt backend/model/sbert_retrieval/1
mv qa/model.pt backend/model/qa_model/1
mv corpus.json backend/model/corpus.json
rm -r sbert
rm -r qa
rm model-compile-v$VERSION.zip
export PYTHONNOUSERSITE=True
pip install conda-pack 
pip uninstall e2eqavn -y
pip install e2eqavn==0.1.9
conda-pack -n dev -o backend/model/dev.tar.gz
mkdir model/ensemble_model/1
