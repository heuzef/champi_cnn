from http.server import CGIHTTPRequestHandler, HTTPServer

# Port sur lequel le serveur écoutera
PORT = 8888

class CustomCGIHandler(CGIHTTPRequestHandler):
    cgi_directories = ['./localhost/cgi-bin','/localhost/cgi-bin','./localhost/cgi-bin','/cgi-bin', '/notebooks/mushroom_observer/localhost/cgi-bin']  # Chemin du répertoire CGI

def run(server_class=HTTPServer, handler_class=CustomCGIHandler, port=PORT):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Serveur démarré sur le port {port}')
    httpd.serve_forever()

if __name__ == "__main__":
    run()