import pyads

print("build target")
target = pyads.Connection('192.168.1.2.1.1', 851)

print("open target")
target.open()


target.write_by_name("MAIN.FL_ActVelo",5.31, pyads.PLCTYPE_LREAL)
i = target.read_by_name("MAIN.FL_ActVelo", pyads.PLCTYPE_LREAL)
print(i)



target.write_by_name("MAIN.FL_ActVelo",2.31, pyads.PLCTYPE_LREAL)
f = target.read_by_name("MAIN.FL_ActVelo", pyads.PLCTYPE_LREAL)
print(f)
