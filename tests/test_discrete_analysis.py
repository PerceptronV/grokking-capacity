from scripts.analyse_discrete import memorisation_intervals


def test_separate_plateaus_and_censoring():
    records = [dict(step=s, train_acc=.995, val_acc=v) for s,v in
               [(0,.95),(100,.5),(200,.6),(300,.8),(400,.91),(500,.7),(600,.8)]]
    assert memorisation_intervals(records) == [
        dict(start_step=100,end_step=300,observed_steps=200,observations=3,end_censored=False),
        dict(start_step=500,end_step=600,observed_steps=100,observations=2,end_censored=True)]


def test_threshold_jitter_is_not_low_heldout_plateau():
    records=[dict(step=i*100,train_acc=t,val_acc=v) for i,(t,v) in enumerate(
        [(.992,.98),(.997,.995),(.98,.97),(.999,.99)])]
    assert memorisation_intervals(records) == []
    assert memorisation_intervals([dict(step=0,train_acc=1.)]) == []
