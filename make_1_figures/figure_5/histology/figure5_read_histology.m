read_path = '/mnt/staging/users/wx2203/data_submission/figures_data/figureS5/';
read_file_path = [read_path 'csn_10x.tif'];

t=Tiff(read_file_path,'r');
imageData = read(t);

%%
imagesc(imageData);
