import seaborn as sns
import matplotlib.pyplot as plt
import os
if not os.path.exists("../Pics"):
   os.makedirs("../Pics")
sns.barplot(x= ["fsfsf","dadada"], y= [43, 434])
plt.savefig("../Pics/output.jpg")
# plt.show()

# sns.barplot(["hhhhh","ccccc"], [43, 434])
# plt.show()

# sns.barplot(["bbbbb","fffff"], [43, 434])
# plt.show()
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

print(color.BLUE + "fsfsfsf")