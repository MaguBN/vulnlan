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
var7 SQL_QUERY var5
var8 mysql_connect
mysql_select_db
echo var7
var9 mysql_query var7
loop var10 mysql_fetch_array var9
  print_r var10
  echo
mysql_close var8
