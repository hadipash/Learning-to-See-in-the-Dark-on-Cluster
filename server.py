import socket
from time import sleep

host = '0.0.0.0'
port = 9999

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))
s.listen(1)
while True:
    print('\nListening for a client at',host , port)
    conn, addr = s.accept()
    print('\nConnected by', addr)
    try:
        print('\nWaiting for user input...\n')
        '''
        with open('iris_test.csv') as f:
            for line in f:
                out = line.encode('utf-8')
                print('Sending line',line)
                conn.send(out)
                sleep(10)
            print('End Of Stream.')
        '''
        while 1:
            sentence = raw_input("What is your sentence? \n")
            sentence = sentence.encode('utf-8')
            conn.send(sentence + '\n')
            #conn.close()
    except:
        print ('Error Occured.\n\nClient disconnected.\n')
conn.close()