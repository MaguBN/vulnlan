var1 array
0 array
1 array
2 array
var2
var3 proc_open var1 var4 var2 NULL
if is_resource var3
  fclose var4_0
  var5 stream_get_contents var4_1
  fclose var4_1
  var6 proc_close var3
class class_name1
  public function function_name1 var7
    return mysql_real_escape_string var7
var8 new class_name1
var5 var8->function_name1 var5
var9 sprintf SQL_QUERY var5
var10 mysql_connect
mysql_select_db
echo var9
var11 mysql_query var9
loop var12 mysql_fetch_array var11
  print_r var12
  echo
mysql_close var10
