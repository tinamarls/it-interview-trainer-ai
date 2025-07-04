import React, { useState, useEffect, useContext } from "react";
import "./Vacancy.css";
import { fetchWithAuth } from "../../api/fetchWithAuth";
import { AuthContext } from "../AuthProvider";
import { useNavigate } from "react-router-dom";

export const Vacancy = () => {
    const [title, setTitle] = useState("");
    const [url, setUrl] = useState("");
    const [vacancies, setVacancies] = useState([]);
    const [errorMessage, setErrorMessage] = useState("");
    const [successMessage, setSuccessMessage] = useState("");
    const [selectedVacancy, setSelectedVacancy] = useState(null);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const { token } = useContext(AuthContext);
    const navigate = useNavigate();

    useEffect(() => {
        if (token) {
            fetchVacancies();
        }
    }, [token]);

    const fetchVacancies = async () => {
        try {
            const response = await fetchWithAuth("/vacancy/all", { method: "GET" }, token);
            const data = await response.json();
            setVacancies(data);
        } catch (error) {
            setErrorMessage("Не удалось загрузить список вакансий.");
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!title.trim() || !url.trim()) {
            setErrorMessage("Заполните все поля.");
            return;
        }

        try {
            const response = await fetchWithAuth("/vacancy/save", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ title, url })
            }, token);

            if (response.ok) {
                setSuccessMessage("Вакансия успешно добавлена!");
                setTitle("");
                setUrl("");
                fetchVacancies();
                setTimeout(() => setSuccessMessage(""), 3000);
            } else {
                setErrorMessage("Ошибка при добавлении вакансии.");
            }
        } catch (error) {
            setErrorMessage("Ошибка при добавлении вакансии.");
        }
    };

    const formatDate = (dateString) => {
        const date = new Date(dateString);
        return date.toLocaleDateString() + " " + date.toLocaleTimeString();
    };

    const openModal = (vacancy) => {
        setSelectedVacancy(vacancy);
        setIsModalOpen(true);
    };

    const closeModal = () => {
        setIsModalOpen(false);
        setSelectedVacancy(null);
    };

    return (
        <div className="vacancy-container">
            {/*<h1 className="page-title">Управление вакансиями</h1>*/}
            <h1 className="page-title">Управление вакансиями</h1>
            <button
                className="compare-button"
                onClick={() => navigate('/compare')} // Переход на страницу сравнения
            >
                Сравнить резюме и вакансию
            </button>

            {/* Форма добавления вакансии */}
            <div className="vacancy-form">
                <h2>Добавить новую вакансию</h2>
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label htmlFor="title">Название:</label>
                        <input
                            type="text"
                            id="title"
                            value={title}
                            onChange={(e) => setTitle(e.target.value)}
                            placeholder="Введите название вакансии"
                        />
                    </div>
                    <div className="form-group">
                        <label htmlFor="url">Ссылка:</label>
                        <input
                            type="url"
                            id="url"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            placeholder="Введите URL вакансии"
                        />
                    </div>
                    <button type="submit">Добавить вакансию</button>
                </form>

                {errorMessage && <div className="error-message">{errorMessage}</div>}
                {successMessage && <div className="success-message">{successMessage}</div>}
            </div>

            {/* Список вакансий */}
            <div className="vacancy-list">
                <h2>Мои вакансии</h2>
                {vacancies.length === 0 ? (
                    <p>Нет сохраненных вакансий</p>
                ) : (
                    <table>
                        <thead>
                        <tr>
                            <th>Название</th>
                            <th>Ссылка</th>
                            <th>Дата добавления</th>
                        </tr>
                        </thead>
                        <tbody>
                        {vacancies.map((vacancy) => (
                            <tr key={vacancy.id}>
                                <td>
                                    <button
                                        className="vacancy-title-button"
                                        onClick={() => openModal(vacancy)}
                                    >
                                        {vacancy.title}
                                    </button>
                                </td>
                                <td>
                                    <a href={vacancy.url} target="_blank" rel="noopener noreferrer">
                                        {vacancy.url}
                                    </a>
                                </td>
                                <td>{formatDate(vacancy.createdAt)}</td>
                            </tr>
                        ))}
                        </tbody>
                    </table>
                )}
            </div>

            {/* Модальное окно */}
            {isModalOpen && selectedVacancy && (
                <div className="modal-overlay" onClick={closeModal}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <button className="modal-close" onClick={closeModal}>×</button>
                        </div>
                        <div className="modal-body">
                            <div className="extracted-text">
                                {selectedVacancy.text || "Текст не найден"}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};