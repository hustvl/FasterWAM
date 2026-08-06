cd assets
python _download.py

# background_texture
unzip background_texture.zip
rm -f background_texture.zip

# embodiments
unzip embodiments.zip
rm -f embodiments.zip

# objects
unzip objects.zip
rm -f objects.zip

cd ..
echo "Configuring Path ..."
python script/update_embodiment_config_path.py
