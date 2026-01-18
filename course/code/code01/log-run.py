from datetime import datetime

myerr = 0.23 # example 
msg = f"{datetime.now()}  rmse={myerr}\n"
with open("log-run.log", "a") as f:
    f.write(msg) # a is for append

with open("log-run.log") as f:
    last_line = f.readlines()[-1]
    print(last_line.strip())
