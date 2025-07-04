import React, { useState, useEffect, useContext } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { useParams } from 'react-router-dom';
import { AuthContext } from '../AuthProvider';
import { Container, Spinner, Button, Row, Col } from 'react-bootstrap';
import './PDFViewer.css';

// Указываем путь к worker из установленного пакета
pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js`;
// Или используйте локальный файл:
// import pdfjsWorker from 'pdfjs-dist/build/pdf.worker.entry';
// pdfjs.GlobalWorkerOptions.workerSrc = pdfjsWorker;

export const PDFViewer = () => {
    const [numPages, setNumPages] = useState(null);
    const [pageNumber, setPageNumber] = useState(1);
    const [loading, setLoading] = useState(true);
    const [pdfData, setPdfData] = useState(null);
    const [error, setError] = useState(null);
    const { resumeId } = useParams();
    const { token } = useContext(AuthContext);

    useEffect(() => {
        const fetchPdf = async () => {
            if (!resumeId || !token) return;

            try {
                setLoading(true);
                setError(null);

                const response = await fetch(`http://localhost:8081/resume/view/${resumeId}`, {
                    method: 'GET',
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                });

                if (!response.ok) {
                    throw new Error(`Ошибка при загрузке PDF: ${response.status}`);
                }

                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                setPdfData(url);
            } catch (err) {
                console.error('Ошибка при загрузке PDF:', err);
                setError(`Не удалось загрузить PDF: ${err.message}`);
            } finally {
                setLoading(false);
            }
        };

        fetchPdf();

        // Очистка URL при размонтировании компонента
        return () => {
            if (pdfData) {
                URL.revokeObjectURL(pdfData);
            }
        };
    }, [resumeId, token]);

    function onDocumentLoadSuccess({ numPages }) {
        setNumPages(numPages);
        setPageNumber(1);
    }

    function changePage(offset) {
        setPageNumber(prevPageNumber => Math.min(Math.max(prevPageNumber + offset, 1), numPages));
    }

    function previousPage() {
        changePage(-1);
    }

    function nextPage() {
        changePage(1);
    }

    return (
        <Container className="pdf-viewer-container mt-4">
            <h2 className="text-center mb-4">Просмотр резюме</h2>

            {loading && (
                <div className="text-center">
                    <Spinner animation="border" role="status">
                        <span className="visually-hidden">Загрузка...</span>
                    </Spinner>
                    <p className="mt-2">Загрузка PDF...</p>
                </div>
            )}

            {error && (
                <div className="alert alert-danger" role="alert">
                    {error}
                </div>
            )}

            {pdfData && (
                <div className="pdf-document-container">
                    <Document
                        file={pdfData}
                        onLoadSuccess={onDocumentLoadSuccess}
                        onLoadError={(error) => setError(`Ошибка при открытии PDF: ${error.message}`)}
                        loading={<Spinner animation="border" />}
                    >
                        <Page
                            pageNumber={pageNumber}
                            renderTextLayer={false}
                            renderAnnotationLayer={false}
                            width={Math.min(800, window.innerWidth - 30)}
                        />
                    </Document>

                    {numPages && (
                        <Row className="pagination-controls mt-3">
                            <Col className="text-center">
                                <Button
                                    onClick={previousPage}
                                    disabled={pageNumber <= 1}
                                    variant="outline-primary"
                                    className="me-2"
                                >
                                    Предыдущая
                                </Button>
                                <span className="page-info">
                                    Страница {pageNumber} из {numPages}
                                </span>
                                <Button
                                    onClick={nextPage}
                                    disabled={pageNumber >= numPages}
                                    variant="outline-primary"
                                    className="ms-2"
                                >
                                    Следующая
                                </Button>
                            </Col>
                        </Row>
                    )}
                </div>
            )}
        </Container>
    );
};

export default PDFViewer;