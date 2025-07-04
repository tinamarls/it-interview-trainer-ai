import React, { useState, useEffect, useContext } from "react";
import "./Resume.css";
import "../../index.css";
import { fetchWithAuth } from "../../api/fetchWithAuth";
import { AuthContext } from "../AuthProvider";
import { useNavigate } from "react-router-dom";
import { Card, Button, Container, Row, Col, Form, Alert } from "react-bootstrap";

export const Resume = () => {
    const [resumes, setResumes] = useState([]);
    const [errorMessage, setErrorMessage] = useState("");
    const [file, setFile] = useState(null);
    const [fileName, setFileName] = useState("");
    const { token } = useContext(AuthContext);
    const navigate = useNavigate();

    const fetchResumes = async () => {
        try {
            const response = await fetchWithAuth("/resume/all", {method: "GET"}, token);
            const data = await response.json();
            setResumes(data);
        } catch (error) {
            setErrorMessage("Не удалось загрузить список резюме.");
        }
    };

    useEffect(() => {
        if (token) {
            fetchResumes();
        }
    }, [token]);

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        setFile(selectedFile);
        setFileName(selectedFile?.name || "");
    };

    const handleUpload = async () => {
        if (!file) {
            setErrorMessage("Выберите файл для загрузки.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetchWithAuth("/resume/upload", { method: "POST", body: formData }, token);
            setFile(null);
            setFileName("");

            if (response.ok) {
                fetchResumes(); // Обновим список без перезагрузки страницы
            } else {
                const errorText = await response.text();
                setErrorMessage(`Ошибка при загрузке: ${errorText}`);
            }
        } catch (error) {
            setErrorMessage("Произошла ошибка при загрузке файла.");
        }
    };

    const handleDownload = async (resumeId, fileName) => {
        try {
            const response = await fetch(`http://localhost:8081/resume/download/${resumeId}`, {
                method: "GET",
                headers: {
                    Authorization: `Bearer ${token}`,
                },
            });

            if (!response.ok) {
                throw new Error("Ошибка при загрузке файла");
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);

            const link = document.createElement("a");
            link.href = url;
            link.download = fileName || "resume.pdf";
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error("Ошибка при скачивании файла:", error);
            setErrorMessage("Не удалось скачать файл.");
        }
    };

    return (
        <Container fluid className="resume-page-wide mt-5">
            <div className="resume-content">
                {/* Заголовок и кнопка сравнения */}
                <Row className="mb-4">
                    <Col>
                        <h1 className="page-title">Мои резюме</h1>
                    </Col>
                </Row>

                <Row className="mb-4">
                    <Col>
                        <Button
                            variant="primary"
                            className="compare-button"
                            onClick={() => navigate('/compare')}
                        >
                            Сравнить резюме и вакансию
                        </Button>
                    </Col>
                </Row>

                {/* Форма добавления нового резюме */}
                <Card className="upload-card mb-4">
                    <Card.Body className="upload-card-body">
                        <Card.Title className="upload-card-title">Добавить новое резюме</Card.Title>
                        <Form className="resume-upload-form">
                            <Form.Group className="mb-3 file-upload-group">
                                <Form.Label className="file-upload-label">Выбрать файл</Form.Label>
                                <Form.Control
                                    type="file"
                                    onChange={handleFileChange}
                                    accept=".pdf,.doc,.docx"
                                    className="form-control-wide"
                                />
                                {fileName && (
                                    <p className="selected-file-name mt-2">Выбран: {fileName}</p>
                                )}
                            </Form.Group>

                            <Button
                                variant="success"
                                onClick={handleUpload}
                                className="upload-button"
                            >
                                Загрузить резюме
                            </Button>
                        </Form>

                        {errorMessage && (
                            <Alert variant="danger" className="mt-2">
                                {errorMessage}
                            </Alert>
                        )}
                    </Card.Body>
                </Card>

                {/* Список резюме */}
                <h2 className="section-title">Загруженные резюме</h2>
                <div className="resume-list-container">
                    {resumes.length === 0 ? (
                        <Alert variant="info">Вы еще не загрузили ни одного резюме.</Alert>
                    ) : (
                        resumes.map((resume, index) => (
                            <Card key={resume.id} className="resume-card mb-3">
                                <Card.Body>
                                    <div className="d-flex justify-content-between align-items-center">
                                        <div className="d-flex align-items-center">
                                            <div className="resume-number">{index + 1}</div>
                                            <div>
                                                <div className="resume-filename">{resume.fileName}</div>
                                                <div className="resume-time">
                                                    {new Date(resume.uploadedAt).toLocaleString()}
                                                </div>
                                            </div>
                                        </div>
                                        <Button
                                            variant="outline-primary"
                                            onClick={() => handleDownload(resume.id, resume.fileName)}
                                            className="download-btn"
                                        >
                                            Скачать
                                        </Button>
                                    </div>
                                </Card.Body>
                            </Card>
                        ))
                    )}
                </div>
            </div>
        </Container>
    );
};