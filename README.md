# DECA-ShortScanner-SMG
Implementation is extremely simple and free:

You need Chrome for this, and you cannot use a school Chromebook(Program will not run).
I intended this purely for Windows(sry Mac Users)
You need a decent IDE(I used PyCharm)


Let's start by creating a project folder in PyCharm and adding a Python file inside.

Inside the Python file, copy and paste all of the code from the app.py

Now, make an account with Groq at https://console.groq.com/home

Once signed in, at the top, you'll see API Keys; click on it.

Then, towards the top right, you'll find the Create API Key. Name anything you want since it does not matter.

Copy the API Key(You will have to make a new Key if you do not copy it right now). Go to your Python file and substitute the fake API key on line 18.



Almost there!

Now, go to your project terminal/PowerShell. Here, we need to install a bunch of libraries so the code runs properly.
Individually copy and paste all of these into there:

pip install selenium

pip install groq

pip install beautifulsoup4

pip install requests

Now, your program should be fully functioning! Good luck with your ICDC endeavors.




