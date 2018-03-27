import pyodbc
import os


def make_table_def(schema):
    return ', '.join(['{0} {1} {2}'.format(x['name'], x['sql_type'], x['sql_qual'] if 'sql_qual' in x else '').strip() for x in schema])

def get_tables(con, **kwargs):
    cnd = None
    if 'query' in kwargs:
        cnd = kwargs['query']
    else:
        cnd = ' and '.join(["{0}={1}".format(k, v if type(v) is not str else "'{}'".format(v)) for (k,v) in kwargs.items()])
    cur = con.execute('select * from INFORMATION_SCHEMA.TABLES where {}'.format(cnd))
    return cur.fetchall()

def exist_table(con, **kwargs):
    return len(get_tables(con, **kwargs)) > 0

def create_table(con, name, commit=True, **kwargs):
    if 'query' in kwargs:
        tdef = kwargs['query']
    else:
        tdef = '({})'.format(', '.join(["{0} {1}".format(k, v) for (k,v) in kwargs.items()]))
    req = 'create table {0} {1}'.format(name, tdef)
    con.execute(req)
    if commit:
        con.commit()
    
def create_table_conditional(con, table, check_if_exist=True, drop_if_exist=False, commit=True, **kwargs):
    if type(table) is str:
        if check_if_exist:
            if exist_table(con, table_name=table):
                if drop_if_exist:
                    drop_table(con, table, commit=False)
                else:
                    return
        create_table(con, table, commit=commit, **kwargs)
    elif type(table) is dict:
        for (k, v) in table.items():
            if type(v) is str:
                create_table_conditional(con, k, check_if_exist=check_if_exist, drop_if_exist=drop_if_exist, commit=False, query=v)
            elif type(v) is dict:
                create_table_conditional(con, k, check_if_exist=check_if_exist, drop_if_exist=drop_if_exist, commit=False, **v)
            else:
                con.rollback()
                raise Exception('create_table_conditional: unexpected dictionary argument')
        if commit:
            con.commit()
    else:
        raise Exception('create_table_conditional: unexpected tabl argument, expected string or dictionary')
        
def drop_table(con, name, commit=True):
    req = 'drop table {}'.format(name)
    con.execute(req)
    if commit:
        con.commit()
    
def insert_table(con, name, data, valnames=None, commit=False):
    if type(data) is tuple:
        ddef = ', '.join(['?']*len(data))
        if valnames is not None:
            con.execute('insert into {0} ({1}) values ({2})'.format(name, ', '.join(valnames), ddef), data)
        else:
            con.execute('insert into {0} values ({1})'.format(name, ddef), data)
        if commit:
            con.commit()
    elif type(data) is list:
        if len(data) == 0:
            return
        types = set([(type(x), len(x)) for x in data])
        if len(types) > 1:
            raise Exception('insert_table: inconsistent tuples in list!')
        type0 = list(types)[0]
        if type0[0] is not tuple:
            raise Exception('insert_table: expected tuples in list!')
        ddef = ', '.join(['?']*type0[1])
        cur = con.cursor()
        
        
        if valnames is not None:
            cur.executemany('insert into {0} ({1}) values ({2})'.format(name, ', '.join(valnames), ddef), data)
        else:
            cur.executemany('insert into {0} values ({1})'.format(name, ddef), data)
        
        if commit:
            con.commit()
    else:
        raise Exception('insert_table: expected tuple or list of tuples!')
    
def create_queue_table(con, name, check_if_exist=True, drop_if_exist=False):
    create_table_conditional(con, name, check_if_exist=check_if_exist, drop_if_exist=drop_if_exist,
            query='(rid int not null unique, queue_pid int, queue_status int, queue_start_dt datetime, queue_finish_dt datetime)')
    
def insert_queue_table(con, name, src, pk_name='rid', commit=False):
    if type(src) is list:
        #insert as if python list
        cur = con.cursor()
        cur.executemany('insert into {0} (rid) values (?)'.format(name), [(x,) for x in src])
        if commit:
            con.commit()
    elif type(src) is str:
        con.execute('insert into {0} select {1} as rid, null, null, null, null from {2}'.format(name, pk_name, src))
        if commit:
            con.commit()
    else:
        raise Exception('insert_queue_table: unexpected argument')
        

def popid_queue_table(con, name, num=1):
    cur = con.execute('''
        with t as (select top {0} * from {1} where queue_pid is null order by rid)
        update t set queue_pid = {2}, queue_start_dt = GETDATE() output inserted.rid
        '''.format(num, name, os.getpid()))
    res = cur.fetchall()
    con.commit()
    if res is None:
        return None
    return [x[0] for x in res]

def update_status_queue_table(con, name, rid, status):
    cur = con.execute('select queue_pid from {0} where rid={1}'.format(name, rid))
    res = cur.fetchall()
    if res is None or len(res) == 0:
        raise Exception('Specified rid is not found in table {0}'.format(name))
    if os.getpid() != res[0][0]:
        raise Exception('Specified rid belongs to another process')
    cur = con.execute('update {0} set queue_status = {1}, queue_finish_dt = GETDATE() where rid = {2}'.format(name, status, rid))
    if cur.rowcount != 1:
        con.rollback()
        raise Exception('Update was expected to affect 1 row!')
    con.commit()
    
def create_log_table(con, name, check_if_exist=True, drop_if_exist=False):
    create_table_conditional(con, name, check_if_exist=check_if_exist, drop_if_exist=drop_if_exist,
                             logid='int identity primary key', rid='int', pid='int', dt='datetime', message='varchar(max)')
    
def push_log_table(con, name, rid, message):
    con.execute('insert {0} (rid, pid, dt, message) values (?, ?, GETDATE(), ?)'.format(name),
                rid, os.getpid(), message)
    con.commit()
    
def log_queue_status_update(con, queue_name, log_name, rid, status, msg = None):
    try:
        update_status_queue_table(con, queue_name, rid, status)
    except Exception as e:
        push_log_table(con, log_name, rid, 'Critical failure -- failed to update status: {0}'.format(e))
    except:
        push_log_table(con, log_name, rid, 'Critical failure -- failed to update status due to unknown exception')
    else:
        if msg is not None:
            push_log_table(con, log_name, rid, msg)
    
def process_one(con, queue_name, log_name, rid, fn_process):
    #null - not started, 0 - started, -1 - failed, 1 - succeed
    log_queue_status_update(con, queue_name, log_name, rid, 0)
    res = None
    try:
        res = fn_process(rid)
    except Exception as e:
        res = None
        log_queue_status_update(con, queue_name, log_name, rid, -1, 'Error: {}'.format(e))
    except:
        res = None
        log_queue_status_update(con, queue_name, log_name, rid, -1, 'Error: <unknown>')
    else:
        log_queue_status_update(con, queue_name, log_name, rid, 1)
    return res
    
def queue_process(con, queue_name, log_name, fn_process, queue_batch_size=1):
    rids = popid_queue_table(con, queue_name, num=queue_batch_size)
    fres = []
    for rid in rids:
        fres.append((rid, process_one(con, queue_name, log_name, rid, fn_process)))
    return fres