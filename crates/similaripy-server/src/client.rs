//! A minimal HTTP/1.1 client over one keep-alive connection (loadgen, tests).

use std::io;
use std::net::SocketAddr;

use axum::http::{Method, Request, StatusCode, header};
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::client::conn::http1::{SendRequest, handshake};
use hyper_util::rt::TokioIo;

pub type Conn = SendRequest<Full<Bytes>>;

/// A connection to `addr`; its I/O runs as a task on the current runtime.
pub async fn connect(addr: SocketAddr) -> io::Result<Conn> {
    let s = tokio::net::TcpStream::connect(addr).await?;
    s.set_nodelay(true)?;
    let (send, conn) = handshake(TokioIo::new(s)).await.map_err(io::Error::other)?;
    tokio::spawn(conn);
    Ok(send)
}

/// One request: (status, body). `body` = None sends a GET.
pub async fn send(
    c: &mut Conn,
    path: &str,
    body: Option<Bytes>,
) -> Result<(StatusCode, Bytes), hyper::Error> {
    c.ready().await?;
    let method = if body.is_some() {
        Method::POST
    } else {
        Method::GET
    };
    let req = Request::builder()
        .method(method)
        .uri(path)
        .header(header::HOST, "localhost")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Full::new(body.unwrap_or_default()))
        .expect("a valid request");
    let res = c.send_request(req).await?;
    let status = res.status();
    Ok((status, res.into_body().collect().await?.to_bytes()))
}

/// The sum of `name`'s samples in Prometheus text (`/metrics`): unlabelled,
/// or over every label set (`{proto="json"}` + `{proto="grpc"}`). None if
/// there is none.
pub fn metric(text: &str, name: &str) -> Option<f64> {
    text.lines()
        .filter_map(|l| {
            let rest = l.strip_prefix(name)?;
            let v = match rest.strip_prefix('{') {
                Some(r) => r.split_once("} ")?.1,
                None => rest.strip_prefix(' ')?,
            };
            v.parse::<f64>().ok()
        })
        .reduce(|a, b| a + b)
}
