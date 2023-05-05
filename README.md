Trained street-fighter using the simplest PPO
* the game environment and training parameters from linyi's street-fighter-ai project (https://github.com/linyiLYi/street-fighter-ai )
* The ppo algorithm references The stable_ Baselines 3  of the openai and seungeunrho's minimalRL(https://github.com/seungeunrho/minimalRL)

### Environment Setup
only run on Gym 0.21, therefore Python 3.8 must be used
#### Create a conda environment named street with Python version 3.8.10
```
conda create -n street python=3.8.10

conda activate street
```
#### Run script to locate gym-retro game folder
```
python play -m 2
```
After the console outputs the folder path, copy it to the file explorer and navigate to the corresponding path. This folder contains the game data files for "Street Fighter II: Special Champion Edition" within gym-retro, including the game ROM file and data configuration files. Copy the `Champion.Level12.RyuVsBison.state`, `data.json`, `metadata.json`, and `scenario.json` files from the `data/` folder of this project into the game data folder, replacing the original files (administrator privileges may be required). The `.state` file is a save state for the game's highest difficulty level, while the three `.json` files are gym-retro configuration files storing game information memory addresses (this project only uses [agent_hp] and [enemy_hp] for reading character health values in real-time).

To run the program, you will also need the game ROM file for "Street Fighter II: Special Champion Edition", which is not provided by gym-retro and must be obtained legally through other means. You can refer to this [link](https://wowroms.com/en/roms/sega-genesis-megadrive/street-fighter-ii-special-champion-edition-europe/26496.html).

Once you have legally obtained the game ROM file, copy it to the aforementioned gym-retro game data folder and rename it to `rom.md`. At this point, the environment setup is complete.

Note: If you want to record videos of the AI agent's gameplay, you will need to install [ffmpeg](https://ffmpeg.org/).

```bash
conda install ffmpeg
```
Note: If you want to create more game states, you can use the following command(For example, saving a state can allow two people to fight against each other)
```
python play -m 3
```

### Training the Model
```
python  play -m 0
```
### Test the Model
```
python  play -m 1
```
