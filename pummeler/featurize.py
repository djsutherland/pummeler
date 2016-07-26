# FIXME

def get_dummies(df, real_means, real_stds, value_counts):
    parts = [((df[real_feats] - real_means) / real_stds).fillna(0)]
    for k in discrete_feats + alloc_flags:
        vc = value_counts[k]
        c = pd.Categorical(df[k], categories=vc.index).codes
        n_codes = len(vc)
        names = ['{}_{}'.format(k, v) for v in vc.index]
        if n_null[k] > 0:
            c = c.copy()
            c[c == -1] = n_codes
            n_codes += 1
            names.append('{}_nan'.format(k))
        bit = np.eye(n_codes).take(c, axis=0)
        parts.append(pd.DataFrame(bit, index=df.index, columns=names))
    return pd.concat(parts, axis=1)

