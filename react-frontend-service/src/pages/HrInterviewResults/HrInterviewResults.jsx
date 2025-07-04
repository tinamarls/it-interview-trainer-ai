import React, { useState, useEffect, useContext } from "react";
import { Link } from "react-router-dom";
import "./HrInterviewResults.css"; // Мы используем стили из Vacancy.css
import { fetchWithAuth } from "../../api/fetchWithAuth";
import { AuthContext } from "../AuthProvider";

export const InterviewResults = () => {
    const [attempts, setAttempts] = useState([]);
    const [loading, setLoading] = useState(true);
    const [errorMessage, setErrorMessage] = useState("");
    const { token } = useContext(AuthContext);

    useEffect(() => {
        fetchAttempts();
    }, [token]);

    const fetchAttempts = async () => {
        try {
            setLoading(true);
            const response = await fetchWithAuth("/hr-interview/results",
                { method: "GET" },
                token
            );

            if (response.ok) {
                const data = await response.json();
                setAttempts(data);
            } else {
                const errorData = await response.json();
                setErrorMessage(`Ошибка при загрузке: ${errorData.error || 'Неизвестная ошибка'}`);
            }
        } catch (error) {
            setErrorMessage(`Произошла ошибка: ${error.message}`);
        } finally {
            setLoading(false);
        }
    };

    // Функция для форматирования даты в читаемом виде
    // Функция для форматирования даты в читаемом виде
    const formatDate = (dateString) => {
        if (!dateString) return "Нет данных";
        const date = new Date(dateString);
        return date.toLocaleString();
    };

    return (
        <div className="vacancy-container">
            <h1 className="page-title">Результаты HR-собеседований</h1>

            {errorMessage && <div className="error-message">{errorMessage}</div>}

            <div className="vacancy-list">
                {loading ? (
                    <p>Загрузка данных...</p>
                ) : attempts.length === 0 ? (
                    <p>У вас еще нет пройденных собеседований.</p>
                ) : (
                    <table>
                        <thead>
                        <tr>
                            <th>№</th>
                            <th>Дата прохождения</th>
                            <th>Статус</th>
                            <th>Действия</th>
                        </tr>
                        </thead>
                        <tbody>
                        {attempts.map((attempt, index) => (
                            <tr key={attempt.id}>
                                <td>{index + 1}</td>
                                <td>{formatDate(attempt.endTime)}</td>
                                <td>{attempt.status}</td>
                                <td>
                                    <Link to={`/hr-interview/results/${attempt.id}`}>
                                        Просмотреть результат
                                    </Link>
                                </td>
                            </tr>
                        ))}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    );
};

export default InterviewResults;