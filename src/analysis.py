"""第13讲 购物篮关联规则
业务问题：买A的人是否更常买B？
全部使用模拟教学数据。"""
from pathlib import Path
import sys
if "__file__" in globals():
    COURSE_ROOT = Path(__file__).resolve().parents[1]
else:
    COURSE_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "course_common.py").exists())
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from course_common import (ROOT, DATA, AS_OF, BLUE, TEAL, ORANGE, load_tables,
    build_facts, positive_orders, rfm_table, safe_divide, outdir, save_table, save_figure, show)

from itertools import combinations
from collections import Counter
# %% 每张有效订单每个SKU只计一次，件数不当成不同购物篮
lines,orders=build_facts();valid=set(orders.loc[orders.net_sales_calc.gt(0),'order_id'])
baskets=lines.loc[lines.order_id.isin(valid)&lines.net_units.gt(0)].groupby('order_id').product_id.agg(lambda x:sorted(set(x)))
singles=Counter();pairs=Counter()
for basket in baskets:
    singles.update(basket);pairs.update(combinations(basket,2))
n=len(baskets);rows=[]
for (a,b),count in pairs.items():
    support=count/n;lift=support/((singles[a]/n)*(singles[b]/n))
    rows.append({'A':a,'B':b,'co_orders':count,'support':support,
        'confidence_A_to_B':count/singles[a],'confidence_B_to_A':count/singles[b],'lift':lift})
rules=pd.DataFrame(rows).sort_values(['lift','co_orders'],ascending=False)
selected=rules.loc[rules.co_orders.ge(20)].copy()
show(selected);save_table(rules,13,'all_pair_rules');save_table(selected,13,'rules_min20')
assert rules.support.between(0,1).all() and rules.confidence_A_to_B.between(0,1).all()
top=selected.head(8).sort_values('lift')
fig,ax=plt.subplots(figsize=(8,4));ax.barh(top.A+' + '+top.B,top.lift,color=TEAL);ax.axvline(1,color='gray',ls='--')
ax.set(title='Association lift with at least 20 joint orders',xlabel='Lift (not a causal effect)');save_figure(fig,13,'basket_lift')
print('规则用于提出搭配假设；销量、陈列和促销都可能影响共购，仍需业务验证。')
