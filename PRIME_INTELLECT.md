# Prime Intellect Set-up Guide

Short quick-start guide on how to run scripts through GPU instances using Prime Intellect. 
Before attempting, prepare the scripts you want to run. 

## Setting up UV

Install `uv` on your local computer with the following commands:

For MacOS/Linux:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows:
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Navigate to the folder where your scripts are running and run
```bash
uv init
```
This starts a new Python project and allows you to add dependencies

You can use the following commands to add and remove dependencies from your project as you wish. Commit the files and the configuration will now be stored in GitHub. 
```bash
uv add <pkg>

uv remove <pkg>
``` 

You can run your script using:
```bash
uv run <script>
```
`uv` will automatically read your config and install and include the necessary dependencies,

## Setting up Prime Intellect

You will need to add an ssh key to Prime Intellect. This is what will allow you to actually access the instances. Once you have your ssh key, you can paste it into Prime Intellect. 

## Setting up the instance

1. After opening Prime Intellect, select and instance and rent it. It usually takes a couple of minutes to provision depending on the provider. 

2. Once the instance has been provisioned, it will show an ssh command. 

3. Copy this ssh command and open a new VS Code window. 

4. Use the CTRL+SHIFT+P shortcut to open the Command Palette and search for `Remote SSH: Connect Current Window to Host...`

5. Once you've clicked on it, choose the option to `Add new SSH Host...`

6. Once you've clicked on it, you can copy in the ssh command from earlier into the prompt and press enter. This will add the host to your config. If you follow step 4 again, you should now be able to see the new IP as an option for you to connect to. 

7. Connect to the instance. You might get a popup at the top of your VS code with a message asking you to confirm your connection to an new/unknown host. If so, just type `yes`. 

8. Your VS Code window should now be connected to the Prime Intellect instance. You can open terminal and interact with it there. Once you clone the repo, you can also open the file explorer. Note that you can download anything by right clicking the file in the explorer and choosing the `Download` option. You can also upload anything by dragging it into the explorer. 

## Common Issues

If you set up your ssh key weirdly like me and put it in a different place than usual, you might need to open the config after completing step 6 and add a line `IdentityFile /path/to/key` under the new host. You can access the config file by following step 4 again but instead selecting `Configure SSH Hosts...` and then selecting the correct file.

Sometimes, trying installing `uv` on the instance gives you a permission error that looks something like:
```bash
mkdir: cannot create directory ‘/home/ubuntu/.config/uv’: Permission denied ERROR: unable to create receipt directory at /home/ubuntu/.config/uv
```
You can get around this by running the following command and then attempting the install again. 
```bash
sudo chown -R ubuntu:ubuntu /home/ubuntu/.config
```
