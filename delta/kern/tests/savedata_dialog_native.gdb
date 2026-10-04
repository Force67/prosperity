set pagination off
set confirm off
handle SIGSEGV nostop noprint pass
handle SIGUSR1 nostop noprint pass
handle SIGUSR2 nostop noprint pass
break kern::DceDevice::Ioctl
condition 1 $rsi == 0xc0488204
ignore 1 30
run
python
import gdb, struct
inferior = gdb.selected_inferior()
gdb.execute('disable 1')
buf = int(gdb.parse_and_eval('(void*)malloc(4096)'))
inferior.write_memory(buf, b'/system/common/lib/libSceSaveDataDialog.sprx\x00')
kernel = 35184372088832
module = int(gdb.parse_and_eval(f'((int(*)(void*,unsigned long,void*,unsigned int,void*,void*)){kernel + 178944})((void*){buf},0,0,0,0,(void*)({buf}+4080))'))
assert module > 0, hex(module & 4294967295)
funcs = {}
for name in ['sceSaveDataDialogInitialize', 'sceSaveDataDialogGetStatus', 'sceSaveDataDialogUpdateStatus', 'sceSaveDataDialogTerminate', 'sceSaveDataDialogOpen', 'sceSaveDataDialogClose', 'sceSaveDataDialogGetResult', 'sceSaveDataDialogProgressBarSetValue', 'sceSaveDataDialogProgressBarInc', 'sceSaveDataDialogIsReadyToDisplay']:
    inferior.write_memory(buf, name.encode() + b'\x00')
    rc = int(gdb.parse_and_eval(f'((int(*)(int,void*,void*)){kernel + 182000})({module},(void*){buf},(void*)({buf}+4000))'))
    assert rc == 0, (name, rc)
    funcs[name] = int(gdb.parse_and_eval(f'*(unsigned long*)({buf}+4000)'))

def call(name, arg=0, arg1=0):
    return int(gdb.parse_and_eval(f'((int(*)(unsigned long,unsigned long)){funcs[name]})({arg},{arg1})'))
for cycle in range(700):
    mode = cycle % 7 + 1
    assert call('sceSaveDataDialogInitialize') == 0, ('init', cycle)
    assert call('sceSaveDataDialogGetStatus') == 1
    param = bytearray(152)
    struct.pack_into('<Q', param, 0, 48)
    struct.pack_into('<I', param, 44, buf + 128 + 3234963721 & 4294967295)
    struct.pack_into('<III', param, 48, 152, mode, 1)
    struct.pack_into('<Q', param, 72, buf + 700)
    if mode in (2, 3, 4, 5, 6, 7):
        struct.pack_into('<Q', param, {2: 80, 3: 88, 4: 96, 5: 104, 6: 128, 7: 128}[mode], buf + 400)
    user_data = 1311768464867721216 + cycle
    struct.pack_into('<Q', param, 112, user_data)
    inferior.write_memory(buf + 128, param)
    inferior.write_memory(buf + 400, struct.pack('<IIQII', 0, 0, buf + 850, 0, 0) + bytes(32))
    if mode == 3:
        inferior.write_memory(buf + 400, struct.pack('<I', 2) + bytes(44))
    if mode == 4:
        inferior.write_memory(buf + 400, struct.pack('<I', 2147614725) + bytes(44))
    if mode in (6, 7):
        inferior.write_memory(buf + 400, struct.pack('<I', 1) + bytes(44))
    inferior.write_memory(buf + 850, b'Saving\x00')
    inferior.write_memory(buf + 700, struct.pack('<IIQQI', 1, 0, 0, buf + 900, 1) + bytes(68))
    inferior.write_memory(buf + 900, b'LLEDIALOG' + bytes(23))
    rc = call('sceSaveDataDialogOpen', buf + 128)
    assert rc == 0, ('open', cycle, mode, hex(rc & 4294967295))
    assert call('sceSaveDataDialogIsReadyToDisplay') == 1, ('ready', cycle)
    if mode == 5:
        assert call('sceSaveDataDialogProgressBarSetValue', 0, 150) == 0
        assert call('sceSaveDataDialogProgressBarInc', 0, 5) == 0
        for poll in range(10):
            assert call('sceSaveDataDialogUpdateStatus') == 2, ('running', cycle, poll)
        inferior.write_memory(buf + 600, struct.pack('<I', cycle % 2) + bytes(32))
        assert call('sceSaveDataDialogClose', buf + 600) == 0, ('close', cycle)
    for poll in range(100):
        status = call('sceSaveDataDialogUpdateStatus')
        if status == 3:
            break
        assert status == 2, ('pending close', cycle, status)
        gdb.parse_and_eval('(int)usleep(1000)')
    assert status == 3, ('completion', cycle, status)
    output = bytearray(72)
    struct.pack_into('<Q', output, 16, buf + 2816)
    inferior.write_memory(buf + 2500, bytes([165]) * 16 + output + bytes([165]) * 16)
    inferior.write_memory(buf + 2800, bytes([165]) * 16 + bytes(32) + bytes([165]) * 16)
    assert call('sceSaveDataDialogGetResult', buf + 2516) == 0, ('result', cycle)
    result = bytearray(72)
    canceled = mode == 5 and cycle % 2
    struct.pack_into('<III', result, 0, mode, int(canceled), 0 if canceled else 1)
    struct.pack_into('<Q', result, 16, buf + 2816)
    struct.pack_into('<Q', result, 32, user_data)
    actual = inferior.read_memory(buf + 2500, 104).tobytes()
    assert actual == bytes([165]) * 16 + result + bytes([165]) * 16, ('result fields and guards', cycle, actual.hex())
    expected = b'LLEDIALOG' + bytes(23)
    assert inferior.read_memory(buf + 2800, 64).tobytes() == bytes([165]) * 16 + expected + bytes([165]) * 16, ('selection guards', cycle)
    assert call('sceSaveDataDialogTerminate') == 0, ('term', cycle)
    assert call('sceSaveDataDialogGetStatus') == 0
print('NATIVE_ALL_SEVEN_SAVE_DIALOG_MODES_700_CYCLES_PASSED_NO_OVERRIDES_PROGRESS_CANCEL_GUARDED_RESULTS_SELECTION')
gdb.execute('kill')
end
quit
