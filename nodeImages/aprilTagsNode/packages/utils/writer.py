import yaml

def update_gains(kp,kd,ki,filepath='/code/modcon/packages/solution/GAINS.yaml'):
    with open(filepath,"w") as f:
        gains = {'kp':kp,'kd':kd,'ki':ki}
        yaml.dump(gains,f)
        f.close()

def load_gains(filepath='/code/modcon/packages/solution/GAINS.yaml'):
    with open(filepath,"r") as f:
        gains = yaml.full_load(f)
        f.close()
    return gains['kp'], gains['kd'], gains['ki']
