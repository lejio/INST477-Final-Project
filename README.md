# INST447-Final-Project
---

To run locally follow these instructions:

This assumes you have Python and Git installed on your computer.

The following instructions is for **MacOS** and **Unix** environments only. 

First clone the repository:

Clone the repository
```sh
git clone https://github.com/lejio/INST447-Final-Project
```

```
cd ./INST447-Final-Project
```

Make a environment
```sh
python3 -m venv .venv
```

Activate said environment
```sh
source .venv/bin/activate
```

Install required packages in requirements.txt
```sh
pip install -r requirements.txt
```

Create a ```.env``` file and paste the secrets in there.

Finally, run main.py
```sh
python main.py
```

