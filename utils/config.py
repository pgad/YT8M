''' author: sam tenka
    changed: 2017-05-01
    created: 2017-02-12
    descr: interface for config file 
    usage: the main use is to support other modules. to test this module
           individually, run `python -m utils.config` from `code/`.
            
'''
import glob

def read_config_file(filenm='config.json'):
    try:
        with open(filenm) as f:
            config = eval(f.read())
        return config
    except SyntaxError:
        print('Uh oh... I could not parse the config file. Is it typed correctly? --- utils.config ')
    except IOError:
        print('Uh oh... I could not find the config file. --- utils.config')

config=read_config_file()
def get(attr, root=config):
    ''' Return value of specified configuration attribute. '''
    node = root 
    for part in attr.split('.'):
        node = node[part]
    return node

def print_if_verbose(s):
    if get('VERBOSE'):
        print(s)

def is_file_prefix(attr): 
    ''' Determine whether or not the filename under field `attr` in config.json
        is a prefix of any actual file. For instance, this comes in handy when
        seeing whether or not we have a model checkpoint saved.  
    ''' 
    return bool(glob.glob(get(attr) + '*'))

def test():
    ''' Ensure reading works '''
    assert(get('META.AUTHOR') == 'samtenka')
    assert(get('META.CREATED') == '2017-05-01')
    print('test passed!')

if __name__=='__main__':
    test()

