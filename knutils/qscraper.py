import knutils.qprocess_mssql as qp
import lxml.html as html

class QScraper:
    def fetch_url(con, src_table_name, rid, url_name='url', cnt_interval=(0,)):
        cur = con.execute('select {2} from {0} where rid={1}'.format(src_table_name, rid, url_name))
        urls = cur.fetchall()
        if len(cnt_interval) > 1 and len(urls) > cnt_interval[1]:
            raise Exception('fetch_url received more urls than expected!')
        if len(urls) < cnt_interval[0]:
            raise Exception('fetch_url received less urls than expected!')
        return [x[0] for x in urls]
    
    def parse_url(url):
        return html.parse(url)
    
    def __init__(self, scraper, d_dst_schema):
        std_arg = [
            {'name':'rid', 'sql_type':'int', 'sql_qual':'identity primary key', 'descr':'row-id'},
            {'name':'rdt', 'sql_type':'datetime', 'sql_qual':'not null default GETDATE()', 'descr':'row insert datetime'}
        ]
        self.__scraper = scraper
        self.__map_tables = {k: d_dst_schema[k][0] for k in d_dst_schema}
        self.__map_schema = {k: d_dst_schema[k][1] for k in d_dst_schema}
        self.__def_tables = {d_dst_schema[k][0]:{'query': '({})'.format(qp.make_table_def(std_arg + d_dst_schema[k][1]))}
                             for k in d_dst_schema}
        
    def create_entities(self, con, entity_lst = None, drop_if_exist = False):
        if entity_lst is None:
            qp.create_table_conditional(con, self.__def_tables, drop_if_exist=drop_if_exist, commit=True)
        elif type(entity_lst) is list:
            qp.create_table_conditional(con, {v:self.__def_tables[v] for (k,v) in self.__map_tables.items() if k in entity_lst},
                                        drop_if_exist=drop_if_exist, commit=True)
        else:
            raise Exception('QScraper.create_tables: unsupported agrument type of table_lst')
    
    def drop_entities(self, con, entity_lst = None):
        if entity_lst is None:
            for k in self.__def_tables:
                qp.drop_table(con, k)
        elif type(entity_lst) is list:
            table_lst = [None if x not in self.__map_tables else self.__map_tables[x] for x in entity_lst]
            for k in self.__def_tables:
                if k in table_lst:
                    qp.drop_table(con, k)
        else:
            raise Exception('QScraper.drop_tables: unsupported agrument type of table_lst')
    
    
    def clear_entities(self, con, entity_lst = None):
        if entity_lst is None:
            for k in self.__def_tables:
                con.execute('truncate table {}'.format(k))
            con.commit()
        elif type(entity_lst) is list:
            table_lst = [None if x not in self.__map_tables else self.__map_tables[x] for x in entity_lst]
            for k in self.__def_tables:
                if k in table_lst:
                    con.execute('truncate table {}'.format(k))
            con.commit()
        else:
            raise Exception('QScraper.trunc_tables: unsupported agrument type of table_lst')
    
    def entity_table(self, entity):
        return self.__map_tables[entity]
    
    def save_result(self, con, d_result):
        if len(self.__def_tables)==1 and (type(d_result) is list or type(d_result) is tuple):
            dst = self.__def_tables.keys()[0]
            vals = self.__map_schema.values()[0]
            qp.insert_table(con, dst, d_result, valnames=[x['name'] for x in vals], commit=True)
        elif type(d_result) is dict:
            for (k, data) in d_result.items():
                if k not in self.__map_tables:
                    con.rollback()
                    raise Exception('Result destination not in map of destinations of QScraper')
                qp.insert_table(con, self.__map_tables[k], data, valnames=[x['name'] for x in self.__map_schema[k]], commit=False)
            con.commit()
        else:
            raise Exception('QScraper.save_result: unsupported agrument type of d_result')
            
    def process_one(self, con, src_table, rid, fnproc, url_name='url'):
        urls = QScraper.fetch_url(con, src_table, rid, url_name=url_name, cnt_interval=(0,1))
        if len(urls) == 0:
            return None
        xroot = QScraper.parse_url(urls[0])
        xres = self.__scraper.parse(xroot)
        d_res = fnproc(xres, rid)
        self.save_result(con, d_res)
        return 0
    
    def get_processor(self, con, src_table, fnproc, url_name='url'):
        return lambda x: self.process_one(con, src_table, x, fnproc, url_name=url_name)
    
    def run_process(self, con, q_name, src_name, log_name, fnproc,
                    url_name='url', pk_name='rid', dropq_if_exist=False, dropq_on_complete=False):
        need_fill = True
        if qp.exist_table(con, table_name=q_name):
            need_fill = False
            if dropq_if_exist:
                qp.drop_table(con, q_name, commit=True)
                need_fill = True
        
        if need_fill:
            qp.create_queue_table(con, q_name, drop_if_exist=dropq_if_exist)
            qp.insert_queue_table(con, q_name, src_name, pk_name=pk_name, commit=True)
            
        while True:
            res = qp.queue_process(con, q_name, log_name, self.get_processor(con, src_name, fnproc))
            if res is None or len(res) == 0:
                break
        
        if dropq_on_complete:
            qp.drop_table(con, q_name, commit=True)