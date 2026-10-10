"""Small authored deployment probes; interpret safe expressions, never exec code."""
import ast,json,operator,re


def same(a,b):
    return json.dumps(a,sort_keys=True,ensure_ascii=False,separators=(',',':'))==json.dumps(b,sort_keys=True,ensure_ascii=False,separators=(',',':'))


def expression(node,env):
    if isinstance(node,ast.Constant) and type(node.value) in (int,float,str,bool,type(None)):
        return node.value
    if isinstance(node,ast.Name) and node.id in env:return env[node.id]
    if isinstance(node,ast.UnaryOp):
        fn={ast.USub:operator.neg,ast.UAdd:operator.pos,ast.Not:operator.not_}.get(type(node.op));assert fn
        return fn(expression(node.operand,env))
    if isinstance(node,ast.BinOp):
        a=expression(node.left,env);b=expression(node.right,env)
        fn={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Mod:operator.mod,ast.FloorDiv:operator.floordiv,ast.Div:operator.truediv,ast.Pow:operator.pow,ast.BitAnd:operator.and_}.get(type(node.op));assert fn
        if isinstance(node.op,ast.Pow):assert type(b)==int and 0<=b<=8 and abs(a)<=10000
        return fn(a,b)
    if isinstance(node,ast.Compare):
        a=expression(node.left,env)
        for op,right in zip(node.ops,node.comparators):
            b=expression(right,env);fn={ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,ast.LtE:operator.le,ast.Gt:operator.gt,ast.GtE:operator.ge}.get(type(op));assert fn
            if not fn(a,b):return False
            a=b
        return True
    if isinstance(node,ast.Subscript):
        value=expression(node.value,env);index=expression(node.slice,env);assert type(value) in (list,tuple) and type(index)==int
        return value[index]
    if isinstance(node,ast.IfExp):return expression(node.body if expression(node.test,env) else node.orelse,env)
    raise ValueError('unsupported_expression_AST')


def grade(row,text,ended):
    if not ended:return {'passed':False,'status':'token_limit'}
    value=text.strip();expected=row['expected'];kind=expected['kind']
    try:
        if kind=='text':passed=value==expected['value']
        elif kind=='json':passed=same(json.loads(value),expected['value'])
        elif kind=='python_return':
            assert len(value)<8000
            fence=re.fullmatch(r'```(?:python)?\s*\n(.*?)\n```',value,re.S)
            if fence:value=fence[1]
            tree=ast.parse(value);assert sum(1 for _ in ast.walk(tree))<=128 and len(tree.body)==1
            fn=tree.body[0];assert isinstance(fn,ast.FunctionDef) and fn.name==expected['function'] and not fn.decorator_list
            assert not fn.args.vararg and not fn.args.kwarg and not fn.args.kwonlyargs and not fn.args.defaults and not fn.args.posonlyargs
            body=fn.body
            if len(body)==2 and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):body=body[1:]
            assert len(body)==1 and isinstance(body[0],ast.Return)
            names=[x.arg for x in fn.args.args];assert len(names)==len(set(names))
            passed=True
            for inputs,wanted in expected['cases']:
                assert len(inputs)==len(names);actual=expression(body[0].value,dict(zip(names,inputs)));passed=passed and same(actual,wanted)
        else:raise ValueError('unknown_probe_kind')
        return {'passed':bool(passed),'status':'tested'}
    except (AssertionError,ValueError,TypeError,SyntaxError,IndexError,ZeroDivisionError,OverflowError):
        return {'passed':False,'status':'invalid_format_or_unsupported_safe_AST'}


def evaluate(rows,result):
    assert result['completed'] and result['eos_respected'] and len(rows)==len(result['requests'])==32
    counts={d:{'passed':0,'total':0} for d in ('math','zh','format','code')};items=[]
    for index,(row,answer) in enumerate(zip(rows,result['requests'])):
        assert answer['request_index']==index;g=grade(row,answer['text'],answer['eos_reached']);c=counts[row['domain']];c['total']+=1;c['passed']+=int(g['passed']);items.append({'request_index':index,'probe_id':row['probe_id'],'domain':row['domain'],**g})
    assert all(v['total']==8 for v in counts.values())
    return {'domain_counts':counts,'items':items,'scope':'32 authored short deployment checks; safe code expression subset, not HumanEval or broad independent model quality acceptance.'}


def main():
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True)
    args=parser.parse_args()
    rows=[json.loads(x) for x in args.requests.read_text(encoding='utf-8').splitlines() if x.strip()]
    result=json.loads(args.result.read_text(encoding='utf-8'))
    print(json.dumps(evaluate(rows,result),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
