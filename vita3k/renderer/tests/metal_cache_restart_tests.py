#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Isolated Metal GPU tests; does not launch the emulator. Requires macOS Metal 3."""
from pathlib import Path
import os,subprocess,json,shutil,fcntl,re
import argparse,tempfile
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('binary_dir',type=Path)
parser.add_argument('--vertex',type=Path,help='Optional Sly 25188... GXP vertex fixture')
parser.add_argument('--fragment',type=Path,help='Optional Sly 028582... GXP fragment fixture')
args=parser.parse_args()
if bool(args.vertex) != bool(args.fragment):parser.error('Both GXP fixtures are required together')
workspace=tempfile.TemporaryDirectory(prefix='vita3k-metal-cache-test-')
root=Path(workspace.name)
binary=args.binary_dir.resolve()
results=[]
def run(name, command, extra=None):
 env=dict(os.environ,MTL_DEBUG_LAYER='1');env.update(extra or {})
 p=subprocess.run([str(x) for x in command],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,timeout=120)
 (root/(name+'.log')).write_text(p.stdout)
 result={'name':name,'exit':p.returncode}
 for line in p.stdout.splitlines():
  if line.startswith('CACHE_RESULT '):result['stats']=json.loads(line[len('CACHE_RESULT '):])
  if 'Metal cache: shader hits' in line:
   result['renderer_stats']=line
   result['renderer_counts']=[int(x) for x in re.findall(r'\d+',line.split('Metal cache: ')[1])]
 results.append(result);(root/'test-results.json').write_text(json.dumps(results,indent=2)+'\n');print(result,flush=True)
 assert p.returncode==0,p.stdout[-3000:]
 return result
cache=root/'validation-cache'
if cache.exists():shutil.rmtree(cache)
test=binary/'metal-cache-validation'
run('native-cold',[test,cache,'cold']);run('native-warm',[test,cache,'warm'])
files=sorted(cache.rglob('*.mslcache'))
files[0].write_bytes(b'truncated')
b=bytearray(files[1].read_bytes());b[-1]^=128;files[1].write_bytes(b)
# A damaged magic/header must be rejected before reading metadata.
b=bytearray(files[2].read_bytes());b[0]=0;files[2].write_bytes(b)
archive=next(cache.rglob('*.metalarc'));archive.write_bytes(b'invalid metal archive')
r=run('corrupt-fallback',[test,cache,'fallback']);assert r['stats']['rejected_files']==4
run('corrupt-repaired-warm',[test,cache,'warm'])
with archive.open('r+b') as f:f.truncate(513*1024*1024)
r=run('oversized-archive',[test,cache,'fallback']);assert r['stats']['rejected_files']==1
run('oversized-archive-repaired',[test,cache,'warm'])
with files[0].open('r+b') as f:f.truncate(9*1024*1024)
r=run('oversized-program',[test,cache,'fallback']);assert r['stats']['rejected_files']==1
run('oversized-program-repaired',[test,cache,'warm'])
# Holding the OS lock simulates another live writer; the child must read and render.
with open(archive.parent/'.lock','rb') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 r=run('concurrent-readonly',[test,cache,'readonly']);assert r['stats']['pipeline_hits']==7
block=root/'not-a-directory';block.write_text('cache writes must fail safely')
r=run('unwritable-fallback',[test,block/'cache','fallback']);assert r['stats']['io_errors']>0
disabled=root/'disabled-cache';run('cache-disabled',[test,disabled,'disabled']);assert not disabled.exists()
if args.vertex:
 cmd=[binary/'metal-gxm-validation',args.vertex.resolve(),args.fragment.resolve()]
 gxmcache=root/'gxm-cache'
 if gxmcache.exists():shutil.rmtree(gxmcache)
 run('gxm-cold',cmd,{'VITA3K_METAL_TEST_CACHE':str(gxmcache)})
 r=run('gxm-warm',cmd,{'VITA3K_METAL_TEST_CACHE':str(gxmcache)})
 assert r['renderer_counts']==[23,0,0,27,0,0,0,0],r
 # The existing Delete Shader Caches UI removes precisely this title subtree.
 shutil.rmtree(gxmcache/'shaders/CACHE_TEST')
 run('gxm-after-ui-delete',cmd,{'VITA3K_METAL_TEST_CACHE':str(gxmcache)})
 disabled=root/'gxm-disabled'
 run('gxm-disabled',cmd,{'VITA3K_METAL_TEST_CACHE':str(disabled),'VITA3K_METAL_TEST_CACHE_DISABLED':'1'})
 assert not list(disabled.rglob('*.mslcache')) and not list(disabled.rglob('*.metalarc'))
if (binary/'metal-depth-store-validation').exists():run('depth-readback', [binary/'metal-depth-store-validation'])
print('ALL CHECKS PASSED',flush=True)
