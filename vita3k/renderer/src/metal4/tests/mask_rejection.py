"""Interpret emitted mask predicates and check public shader control flow offline."""
import itertools
import math
import re
import subprocess


def check_module(path):
    samples, enabled, update, fetch, store = map(int, re.findall(r'\d+', path.stem))
    text = subprocess.check_output(['spirv-dis', '--raw-id', str(path)], text=True)
    instructions = []
    names = {}
    for line in text.splitlines():
        match = re.match(r'\s*(?:(%\d+) = )?(Op\w+)\s*(.*)', line)
        if not match:
            continue
        result, opcode, operands = match.groups()
        args = operands.split()
        instructions.append((result, opcode, args))
        if opcode == 'OpName':
            names[operands.split(' ', 1)[1].strip('"')] = args[0]
    loads = {r: a[1] for r, op, a in instructions if op == 'OpLoad'}
    reads = [(r, a) for r, op, a in instructions
             if op == 'OpImageRead' and loads.get(a[1]) == names.get('f_mask')]
    active = enabled and not update
    assert len(reads) == (samples if active else 0), path
    assert ('metal_mask_coverage' in names) == bool(active and samples > 1)
    if not active:
        return 0
    assert not any(op == 'OpExecutionMode' and a[1] == 'EarlyFragmentTests'
                   for _, op, a in instructions), path
    assert any(op == 'OpDecorate' and a == [names['f_mask'], 'NonWritable']
               for _, op, a in instructions)

    # Build main's CFG. Every guest-function call, framebuffer read and interlock
    # begin must be dominated by the accepting edge of the mask decision.
    blocks = {}
    current = None
    in_main = False
    for result, op, args in instructions:
        if op == 'OpFunction':
            in_main = result == names['main_fs']
        if not in_main:
            continue
        if op == 'OpLabel':
            current = result
            blocks[current] = []
        elif current:
            blocks[current].append((result, op, args))
        if op == 'OpFunctionEnd':
            in_main = False
    branch = next((a for b in blocks.values() for _, op, a in b
                   if op == 'OpBranchConditional'))
    predicate, rejected, accepted = branch[:3]
    assert any(op == 'OpKill' for _, op, _ in blocks[rejected])
    effects = {label for label, body in blocks.items() if any(
        op in ('OpFunctionCall', 'OpBeginInvocationInterlockEXT')
        or (op == 'OpImageRead' and loads.get(a[1]) != names['f_mask'])
        for _, op, a in body)}
    assert effects
    reachable = set()
    todo = [next(iter(blocks))]
    while todo:
        label = todo.pop()
        if label == accepted or label in reachable:
            continue
        reachable.add(label)
        for _, op, a in blocks[label]:
            if op == 'OpBranch':
                todo.append(a[0])
            elif op == 'OpBranchConditional':
                todo.extend(a[1:3])
    assert not (effects & reachable), path

    # Ensure cached coverage reaches the sample-mask output in the actual
    # finalizer, without a second mask read or guest-memory store.
    if samples > 1:
        output = names['metal_sample_mask']
        cache = names['metal_mask_coverage']
        cached_loads = {r for r, op, a in instructions if op == 'OpLoad' and a[1] == cache}
        composites = {r for r, op, a in instructions if op == 'OpCompositeConstruct' and a[1:] == list(cached_loads)}
        assert any(op == 'OpStore' and a[0] == output and a[1] in composites
                   for _, op, a in instructions)
    main_msl = path.with_suffix('.metal').read_text().split('fragment ', 1)[1]
    assert main_msl.index('discard_fragment();') < main_msl.index('primary_program(')
    assert 'early_fragment_tests' not in main_msl
    if fetch in (1, 2):
        # Native framebuffer fetch is an input parameter: physical fetch cost
        # is unspecified. Its logical use still follows the rejection.
        assert main_msl.index('discard_fragment();') < main_msl.rindex('last_frag_data')

    constants = {}
    floats = {r for r, op, _ in instructions if op == 'OpTypeFloat'}
    for result, op, args in instructions:
        if op == 'OpConstant':
            constants[result] = float(args[1]) if args[0] in floats else int(args[1])
        elif op == 'OpConstantComposite' and all(a in constants for a in args[1:]):
            constants[result] = [constants[a] for a in args[1:]]
    # Evaluate only the real emitted predicate's dependency slice. No emulator,
    # Metal device, GPU dispatch, or fake successful guest store is involved.
    definitions = {r: (op, a) for r, op, a in instructions if r}
    def evaluate(texels):
        values = constants.copy()
        def value(result):
            if result in values:
                return values[result]
            op, a = definitions[result]
            if op == 'OpImageRead':
                assert loads[a[1]] == names['f_mask']
                answer = texels[value(a[4])] if samples > 1 else texels[0]
            elif op == 'OpFOrdLessThan':
                answer = [x < y for x, y in zip(value(a[1]), value(a[2]))]
            elif op == 'OpAll':
                answer = all(value(a[1]))
            elif op == 'OpSelect':
                answer = value(a[2] if value(a[1]) else a[3])
            elif op == 'OpBitwiseOr':
                answer = value(a[1]) | value(a[2])
            elif op == 'OpIEqual':
                answer = value(a[1]) == value(a[2])
            else:
                raise AssertionError((path, result, op, a))
            values[result] = answer
            return answer
        expected = sum(1 << sample for sample, texel in enumerate(texels)
                       if any(channel >= .5 or math.isnan(channel) for channel in texel))
        assert value(predicate) == (expected == 0)
        if samples > 1:
            stored = next(a[1] for _, op, a in instructions
                          if op == 'OpStore' and a[0] == names['metal_mask_coverage'])
            assert value(stored) == expected
    cases = 0
    for covered in range(1 << samples):
        evaluate([[1. if covered & (1 << i) else 0.] * 4 for i in range(samples)])
        cases += 1
    for selected in range(samples):
        # All channel combinations around the RGBA8 boundary, exact .5, and
        # ordered-compare NaN policy; other samples remain masked.
        for rgba in itertools.product((0., 127/255, .5, 128/255, 1., float('nan')), repeat=4):
            texels = [[0.] * 4 for _ in range(samples)]
            texels[selected] = rgba
            evaluate(texels)
            cases += 1
    return cases
