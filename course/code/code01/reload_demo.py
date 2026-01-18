# reload_demo.py

import importlib
import reload_demo_fkt as mo

def replace_in_file(str1, str2, filename):
    with open(filename, 'r') as file:
        content = file.read()
    content = content.replace(str1, str2)
    with open(filename, 'w') as file:
        file.write(content)

# Call the greet function initially
print(mo.greet("Roland"))  # Expected: Hello Roland

# After modifying reload_demo_fkt.py, reload it
replace_in_file("Hello", "Good Morning", "reload_demo_fkt.py")

importlib.reload(mo)

# Call the updated greet function
print(mo.greet("Roland"))  # Expected: Good Morning Roland

# Restore the original version of the file
replace_in_file("Good Morning", "Hello", "reload_demo_fkt.py")

