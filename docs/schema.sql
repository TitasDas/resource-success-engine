
      CREATE TABLE Tags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE
      );
    

      CREATE TABLE People (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        rate REAL NOT NULL, -- Rate per hour
        start_date DATE NOT NULL,
        end_date DATE,
        working_hours_sunday REAL DEFAULT 0,
        working_hours_monday REAL DEFAULT 8,
        working_hours_tuesday REAL DEFAULT 8,
        working_hours_wedday REAL DEFAULT 8,
        working_hours_thursday REAL DEFAULT 8,
        working_hours_friday REAL DEFAULT 8,
        working_hours_saturday REAL DEFAULT 0
      );
    

      CREATE TABLE PeopleTags (
        person_id INTEGER,
        tag_id INTEGER,
        PRIMARY KEY (person_id, tag_id),
        FOREIGN KEY (person_id) REFERENCES People(id),
        FOREIGN KEY (tag_id) REFERENCES Tags(id)
      );
    

      CREATE TABLE Projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        project_manager_id INTEGER,
        budget REAL NOT NULL,
        status TEXT CHECK( status IN ('Draft', 'Tentative', 'Confirmed', 'Completed', 'Cancelled') ),
        is_active BOOLEAN DEFAULT 1,
        is_successful BOOLEAN DEFAULT NULL,
        start_date DATE NOT NULL,
        end_date DATE,
        FOREIGN KEY (project_manager_id) REFERENCES People(id)
      );
    

      CREATE TABLE ProjectTags (
        project_id INTEGER,
        tag_id INTEGER,
        PRIMARY KEY (project_id, tag_id),
        FOREIGN KEY (project_id) REFERENCES Projects(id),
        FOREIGN KEY (tag_id) REFERENCES Tags(id)
      );
    

      CREATE TABLE Tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER,
        name TEXT NOT NULL,
        FOREIGN KEY (project_id) REFERENCES Projects(id)
      );
    

      CREATE TABLE AllocatedTasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER,
        person_id INTEGER,
        date DATE NOT NULL,
        hours REAL NOT NULL,
        FOREIGN KEY (task_id) REFERENCES Tasks(id),
        FOREIGN KEY (person_id) REFERENCES People(id)
      );
    

      CREATE TABLE LoggedHours (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER,
        task_id INTEGER,
        date DATE NOT NULL,
        hours REAL NOT NULL,
        allocated_task_id INTEGER,
        FOREIGN KEY (person_id) REFERENCES People(id),
        FOREIGN KEY (task_id) REFERENCES Tasks(id),
        FOREIGN KEY (allocated_task_id) REFERENCES AllocatedTasks(id)
      );
    
