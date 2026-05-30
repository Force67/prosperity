'''
PS4 Stub Generator
Copyright (C) Force67 2019

Requires: https://github.com/idc/ps4libdoc/tree/5.05
and the excellent areolib by SocraticBliss, which can be found here:
https://github.com/SocraticBliss/ps4_module_loader/blob/master/ps4_module.py
'''

import os
import json
from datetime import date

nameTable = []

with open('aerolib.csv') as f:
  lineList = f.readlines()

for line in lineList:
    x = line.split()
    nameTable.append((x[0], x[1]))

def areoFind(id):
    for x, y in nameTable:
        if (x == id):
            return y

def emitHeader(out):
    today = date.today()
    # dd/mm/yy
    d1 = today.strftime("%d/%m/%Y")
    out.write('\n// Copyright (C) Force67\n\n// This file was generated on ' + d1 + '\n\n')

#ensure that you have a proper encoding
#on your json files
'''
Get-ChildItem *.txt -Recurse | ForEach-Object {
$content = $_ | Get-Content

Set-Content -PassThru $_.Fullname $content -Encoding ascii -Force}  
'''

#global Symbol name table to avoid name clashes
symStore = []

def libdParse(file, prettyname):

    #print(file)
    with open(file, 'r') as f:
        info = json.load(f)
        try:
            tname = info['shared_object_name']
        except KeyError:
            tname = prettyname

        with open('out\\' + prettyname + '\\' + prettyname + ".h" , 'w') as hpp:
            hpp.write('#pragma once\n')
            emitHeader(hpp)
            hpp.write('#include "../../vmodLinker.h"\n\n#pragma warning(push, 0)\n\n')

            with open('out\\' + prettyname + '\\' + prettyname + "_api.cpp" , 'w') as api:
                emitHeader(api)
                api.write('#include "' + prettyname + '.h"\n\n')
                api.write('static const modules::funcInfo functions[] = {\n')

                mods = info['modules']
                for mod in mods:
                    for lib in mod['libraries']:
                        if lib['is_export'] == True:

                            with open('out\\' + prettyname + '\\' + lib['name'] + ".cpp" , 'w') as cppx:
                                emitHeader(cppx)
                                cppx.write('#include "../../vmodLinker.h"\n\n')

                                for sym in lib['symbols']:
                                    symname = 'NONAME'
                                    try:
                                        symname = sym['name']
                                        if symname is None:
                                            symname = areoFind(sym['encoded_id'])

                                            #still nothing?
                                            if symname == None:
                                                symname = 'unk_' + sym['encoded_id']
                                            
                                    except KeyError:
                                        symname = areoFind(sym['encoded_id'])
                                        if symname == None:
                                            symname = 'unk_' + sym['encoded_id']

                                    if symname == None:
                                        symname = 'unk_' + sym['encoded_id']

                                    #sanitize the name
                                    symname = symname.replace('-', '_')
                                    symname = symname.replace('+', '_')
                                    #symname = symname.replace('__', 'f__', 1)

                                    if prettyname == 'libc':
                                        symname = 'libc_' + symname

                                    if prettyname == 'libkernel':
                                        symname = 'lk_' + symname

                                    hit = False
                                    counter = 0
                                    for x, y in symStore:
                                        if x == symname:
                                            #add 1 to make it "human readable"
                                            symname = symname + str(y + 1)
                                            symStore[counter][1] = y + 1
                                            hit = True
                                            break
                                        counter += 1

                                    if hit == False:
                                        symStore.append([symname, 0])

                                    hpp.write('int PS4ABI ' + symname + '();\n')
                                    cppx.write('int ' + symname + '()\n{\n	LOG_UNIMPLEMENTED;\n	return 0;\n}\n\n')
                                    api.write('	{0x' + sym['hex_id'] + ', (void*)&' + symname + '},\n')

                api.write('};\n\nMODULE_INIT(' + prettyname + ');\n')
                hpp.write('\n#pragma warning(pop)')


try:
    os.mkdir('out')
except FileExistsError:
    pass

for root, dirs, files in os.walk("system/common/lib"):
    for file in files:
        #.net app
        pos = file.find('.dll')
        if pos != -1:
            continue

        pos = file.find('.self')
        if pos != -1:
            continue

        pos = file.find('.elf')
        if pos != -1:
             continue

        pos = file.find('.sprx')
        fshort = file[0:pos]

        try:
            os.mkdir('out\\' + fshort)
        except FileExistsError:
            pass

        libdParse(root + '/' + file, fshort)


