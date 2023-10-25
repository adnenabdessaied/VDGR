cd data
# Exract the graphs
tar xvfz history_adj_matrices.tar.gz 
tar xvfz question_adj_matrices.tar.gz 
tar xvfz img_adj_matrices.tar.gz 

# Remove the .tar files
rm *.tar.gz

# Download the preprocessed image features
mkdir visdial_img_feat.lmdb
wget https://s3.amazonaws.com/visdial-bert/data/visdial_image_feats.lmdb/data.mdb -O visdial_img_feat.lmdb/data.mdb
wget https://s3.amazonaws.com/visdial-bert/data/visdial_image_feats.lmdb/lock.mdb -O visdial_img_feat.lmdb/lock.mdb

echo Data setup successfully...

cd ..

